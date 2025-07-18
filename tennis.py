# import numpy as np
# import pandas as pd

# df=pd.read_csv("dataset.csv")
#  # Lower the age the more is their fitness
#  # Number of wimbledon wins gives their experience
#  # GrandSlams gives their performance
#  # Lower the ranking better the performance

# X=df[["age","wim wins","grandslams","ranking"]].values.astype(float)
# #Normalization
# X[:, 0] = 1 - (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
# X[:,1]=(X[:, 1] - X[:, 1].min()) / (X[:,1 ].max() - X[:, 1].min())
# X[:,2]=(X[:, 2] - X[:, 2].min()) / (X[:,2 ].max() - X[:, 2].min())
# X[:, 3] = 1 - (X[:, 3] - X[:, 3].min()) / (X[:, 3].max() - X[:, 3].min())


# X = np.hstack((np.ones((X.shape[0], 1)), X))
# y=np.zeros((len(X),1))
# y[1]=1


# def sigmoid(a):
#     return 1 / (1 + np.exp(-a))

# def logreg(X,y,lr=0.01,epochs=100000):
#     a,b=X.shape
#     theta=np.zeros((b,1))
#     for epochs in range(epochs):
#         z=X @ theta
#         y_hat=sigmoid(z)
#         grad=(X.T @ (y_hat-y))/a
#         theta -= lr*grad
#     return theta
# theta=logreg(X,y)

# prob=sigmoid(X @ theta)

# df["Probability"]=prob

# df_sorted=df.sort_values(by="Probability",ascending=False)
# print(df_sorted[["Player","age","ranking","grandslams","wim wins","Probability"]])   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with prediction results
df = pd.read_csv("dataset.csv")

# Recompute features (same normalization as before)
X = df[["age", "wim wins", "grandslams", "ranking"]].values.astype(float)
X[:, 0] = 1 - (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())  # age (fitness)
X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())      # wim wins
X[:, 2] = (X[:, 2] - X[:, 2].min()) / (X[:, 2].max() - X[:, 2].min())      # grandslams
X[:, 3] = 1 - (X[:, 3] - X[:, 3].min()) / (X[:, 3].max() - X[:, 3].min())  # ranking

X = np.hstack((np.ones((X.shape[0], 1)), X))

# Logistic regression functions (same as before)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    for epoch in range(epochs):
        z = X @ theta
        y_hat = sigmoid(z)
        grad = (X.T @ (y_hat - y)) / m
        theta -= lr * grad
    return theta

# Dummy winner target (Alcaraz at index 1)
y = np.zeros((len(X), 1))
y[1] = 1

# Train and predict
theta = train_logistic(X, y)
probs = sigmoid(X @ theta)
df["PredictedProbability"] = probs

# ========== VISUALIZATIONS ==========

# 1. Bar Plot – Predicted probabilities
plt.figure(figsize=(12, 6))
sns.barplot(x="Player", y="PredictedProbability", data=df.sort_values(by="PredictedProbability", ascending=False), palette="viridis")
plt.title("Predicted Probability of Winning Wimbledon 2025")
plt.ylabel("Probability")
plt.xlabel("Player")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Scatter Plot – Age vs. Wimbledon wins
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df["age"], df["wim wins"], c=df["PredictedProbability"], cmap="coolwarm", s=100, edgecolors='k')
plt.title("Age vs Wimbledon Wins Colored by Win Probability")
plt.xlabel("Age")
plt.ylabel("Wimbledon Wins")
plt.colorbar(scatter, label="Predicted Win Probability")
for i, name in enumerate(df["Player"]):
    plt.text(df["age"][i] + 0.2, df["wim wins"][i], name, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Heatmap – Feature correlation
plt.figure(figsize=(8, 5))
sns.heatmap(df[["age", "ranking", "grandslams", "wim wins", "PredictedProbability"]].corr(), annot=True, cmap="Blues")
plt.title("Correlation Between Features")
plt.tight_layout()
plt.show()
