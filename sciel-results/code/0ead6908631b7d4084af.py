"""Symbolic regression with hybrid trig+poly basis and ridge regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before 330s kill
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize for numerical stability
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    def make_features(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        cols = [np.ones(len(Xr))]
        # Linear terms
        for j in range(d):
            cols.append(Xrn[:, j])
        # Quadratic (including cross)
        for i in range(d):
            for j in range(i, d):
                cols.append(Xrn[:, i] * Xrn[:, j])
        # Cubic pure terms
        for j in range(d):
            cols.append(Xrn[:, j] ** 3)
        # Sine/cosine with a few frequencies on each input
        for freq in [0.5, 1.0, 2.0]:
            for j in range(d):
                cols.append(np.sin(freq * Xrn[:, j]))
                cols.append(np.cos(freq * Xrn[:, j]))
        # Cross sin terms (pairwise) for low-d
        if d <= 3:
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append(np.sin(1.0 * Xrn[:, i]) * np.cos(1.0 * Xrn[:, j]))
                    cols.append(np.cos(1.0 * Xrn[:, i]) * np.sin(1.0 * Xrn[:, j]))
        return np.column_stack(cols)

    F = make_features(X)
    # Ridge regression via normal equations with a few lambda values, pick best train fit
    y_mean = y.mean()
    y_centered = y - y_mean
    best_w = None
    best_score = np.inf
    for lam in [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]:
        A = F.T @ F + lam * np.eye(F.shape[1])
        b = F.T @ y_centered
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(F, y_centered, rcond=None)[0]
        pred = F @ w + y_mean
        score = np.mean((pred - y) ** 2)
        if score < best_score:
            best_score = score
            best_w = w

    if best_w is None:
        # fallback: mean predictor
        return lambda Xraw: np.full(len(np.asarray(Xraw, dtype=np.float64)), y_mean)

    def predict(Xraw):
        Fr = make_features(Xraw)
        return Fr @ best_w + y_mean

    return predict
# EVOLVE-BLOCK-END
