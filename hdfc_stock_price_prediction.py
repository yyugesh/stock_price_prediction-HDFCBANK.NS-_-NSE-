# Import Libraries

import warnings
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# Data Collection:

# Download historical stock price for HDFCBANK.NS
hdfc_sp = yf.download("HDFCBANK.NS",
                      "2018-01-01",
                      "2024-12-31")

print(hdfc_sp.head())                     # First 5 rows
hdfc_sp = hdfc_sp[['Close']]              # Choose the particular "Close" column from the dataset
hdfc_sp.dropna(inplace=True)              # Drop rows with missing values
print(hdfc_sp.head())

# Plot the Closing Price

plt.figure(figsize=(12,5))
plt.plot(hdfc_sp['Close'])                # Pick the 'Close' column from the dataset.
plt.title("HDFCBANK.NS Closing Price")
plt.show()

# Rolling Statistics

hdfc_sp['Rolling_Menu'] = hdfc_sp['Close'].rolling(window=30).mean()
plt.figure(figsize=(12,5))
plt.plot(hdfc_sp[["Close", "Rolling_Menu"]])
plt.title("Rolling Mean")
plt.show()

'''
hdfc_sp['Close']   : Accesses the "Close" column of the stock price dataset
.rolling(window=30): It creates a rolling window of size 30, meaning it looks at the last 30 rows (days).
.mean()            : Calculates the mean (average) values in 30-day window.
Rolling_Menu       : Contains the 30-day moving average of the closing price.
'''

# Augmented Dickey-Fuller (ADF)  test
        # to check if a time series is stationary —
        # a critical step in time series forecasting models like ARIMA/SARIMA.

df_result = adfuller(hdfc_sp['Close'])     # hdfc_sp['Close']: Selects the "close" price from the HDFC Bank dataset.
                                           # adfuller(...): This function (from statsmodels.tsa.stattools)
                                                # runs the Augmented Dickey-Fuller test,
                                                # which helps to determine if a time series has a unit root,
                                                # i.e., non-stationary.
print(f"ADF Statistics: {df_result[0]}")   # prints the ADF test statistic, more negative the value,
                                           # the more likely the series is stationary.
print(f"p-value: {df_result[1]}")          # prints the p-value from the test.

'''
If p-value < 0.05:    Reject the null hypothesis (i.e., the series is stationary).

If p-value > 0.05:    Fail to reject the null — the series is non-stationary (you may need to difference it).
'''

# First order differencing
hdfc_sp['Close_diff'] = hdfc_sp['Close'].diff().dropna()
#                       hdfc_sp['Close']: Refers to the column containing the closing stock price.
#                       .diff(): Calculates the first difference of the time series. That means:
#                           Close_diff[t]=Close[t]−Close[t−1]
#                       This removes trend and helps make the series stationary.
hdfc_sp['Close_diff'].dropna().plot(title="Differenced Close Price", figsize=(12,5))

plt.show()

# Train-test split

train = hdfc_sp['Close'][:-90]     # Train on the past (training data).
test = hdfc_sp['Close'][-90:]      # Test on the future (testing data).

'''
[:-90] - all values except the last 90
[-90:] - only the last 90 values.
 '''

# ARIMA model (p=1,d=1,q=1 as an example)

modal_arima = ARIMA(train, order=(1,1,1))    # ARIMA() is a function from statsmodels.tsa.arima.model
                                             # order=(1,1,1) represents the ARIMA hyperparameters
modal_arima = modal_arima.fit()              # fits and (trains) the ARIMA model with the best parameters
                                                # using Maximum Likelihood Estimation.
arima_pred = modal_arima.forecast(steps=90)  # forecasts the next x future values beyond the training set.

# ARIMA Visualization

plt.figure(figsize=(12,5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, arima_pred, label='ARIMA Forecast')
plt.title('ARIMA Forecast vs Actual')
plt.legend()
plt.show()

# SARIMA (p,d,q)x(P,D,Q,s)
model_sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
'''
order=(1,1,1) — the non-seasonal part
seasonal_order=(1,1,1,12) — the seasonal part
SARIMA is better than ARIMA when your time series has seasonality, like monthly or yearly patterns.
'''
model_sarima = model_sarima.fit()              # fits and (trains) the SARIMA model with the best parameters
                                                # using Maximum Likelihood Estimation.
sarima_pred = model_sarima.forecast(steps=90)  # steps=90: Tells the model to look ahead & predict 90 future values.
                                               # These predictions can be compared with actual values in the test set

# SARIMA Visualization

plt.figure(figsize=(12,5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, sarima_pred, label='SARIMA Forecast')
plt.title('SARIMA Forecast vs Actual')
plt.legend()
plt.grid()
plt.show()

# Evaluate and Matrix

def evaluate(y_true, y_pred, name="Model"):
    print(f"{name} Evaluation:")
    print(f"R²: {r2_score(y_true, y_pred):.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print("-" * 42)

evaluate(test, arima_pred, "ARIMA")
evaluate(test, sarima_pred, "SARIMA")