"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations_with_replacement


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 280.0  # safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    def build_features(X):
        cols = [np.ones(len(X))]
        # Raw features
        for j in range(d):
            cols.append(X[:, j])
        # Squared features
        for j in range(d):
            cols.append(X[:, j] ** 2)
        # Cubic features
        for j in range(d):
            cols.append(X[:, j] ** 3)
        # Pairwise products
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
        # Triple products (only if d >= 3)
        if d >= 3:
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
        # Non-linear transforms of individual features
        for j in range(d):
            xj = X[:, j]
            cols.append(np.sin(xj))
            cols.append(np.cos(xj))
            cols.append(np.exp(-xj))
            cols.append(np.log1p(xj))
            cols.append(np.sqrt(xj))
            cols.append(1.0 / (1.0 + xj))
        # Interactions between non-linear features (sin/cos of different vars)
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append(np.sin(X[:, i]) * X[:, j])
                    cols.append(np.cos(X[:, i]) * X[:, j])
                    cols.append(X[:, i] * np.exp(-X[:, j]))
        return np.column_stack(cols)

    F_train = build_features(X)
    # Remove any non-finite columns
    finite_mask = np.all(np.isfinite(F_train), axis=0)
    F_train = F_train[:, finite_mask]

    # Ridge regression for stability
    alpha = 1e-6
    w = np.linalg.solve(F_train.T @ F_train + alpha * np.eye(F_train.shape[1]),
                        F_train.T @ y)

    def predict(X):
        F = build_features(X)
        # Apply same finite mask
        F = F[:, finite_mask]
        return F @ w

    return predict
# EVOLVE-BLOCK-END
