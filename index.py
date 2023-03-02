
import pandas as pd
import yfinance as yf

import datetime
from datetime import date, timedelta
from autots import AutoTS
import plotly.graph_objects as go
today = date.today()

coin = input("Enter Cryptocurrency name: ");
search  = coin.upper() + "-USD"


range = int(input("Enter number of days to predict closing prices for: "))

 

print(f"*******************Fetching Data from API for {search}*******************")
data = yf.download(search,
                      start='2022-10-01',
                      end='2022-11-26',
                      progress=True)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
# print(data.head())
figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], 
                                        high=data["High"], 
                                        low=data["Low"], 
                                        close=data["Close"])])
figure.update_layout(title = f"{coin} Price Analysis", 
                     xaxis_rangeslider_visible=False)
figure.show()

model = AutoTS(forecast_length=range, frequency='infer', ensemble='simple')
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(f"{coin} Price Prediction for the next {range} days:")
print(forecast)
