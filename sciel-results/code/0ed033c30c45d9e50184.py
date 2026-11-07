"""Improved symbolic regression with rich basis and ridge regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _build_features(X):
    """Build a rich set of basis functions for 1D input."""
    d = X.shape[1]
    n = len(X)
    cols = [np.ones(n)]
    
    if d == 1:
        x = X[:, 0]
        # Polynomial terms up to degree 6
        for k in range(1, 7):
            cols.append(x ** k)
        # Trigonometric terms
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2 * x))
        cols.append(np.cos(2 * x))
        # Exponential terms (clipped for stability)
        cols.append(np.exp(np.clip(x, -5, 5)))
        # Log terms (only where x > 0)
        log_x = np.where(x > 1e-8, np.log(np.maximum(x, 1e-8)), 0.0)
        cols.append(log_x)
        # Absolute value and sign
        cols.append(np.abs(x))
        # Scaled versions for different ranges
        cols.append(np.tanh(x))
    else:
        # For multi-dimensional (unlikely per task spec, but handle gracefully)
        for j in range(d):
            cols.append(X[:, j])
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
    
    return np.column_stack(cols)


def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    Xf_train = _build_features(X_train)
    n_features = Xf_train.shape[1]
    
    # Use ridge regression with small lambda for stability
    # Try a few lambda values and pick best by training MSE
    best_w = None
    best_mse = np.inf
    
    lambdas = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    for lam in lambdas:
        # Ridge: (X^T X + lam*I)^{-1} X^T y
        try:
            A = Xf_train.T @ Xf_train
            b = Xf_train.T @ y_train
            reg = A + lam * np.eye(n_features)
            w = np.linalg.solve(reg, b)
            pred = Xf_train @ w
            mse = np.mean((pred - y_train) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_w = w
        except np.linalg.LinAlgError:
            continue
    
    if best_w is None:
        # Fallback to lstsq
        best_w, *_ = np.linalg.lstsq(Xf_train, y_train, rcond=None)
    
    w_final = best_w
    
    def predict(X):
        Xf = _build_features(X)
        return Xf @ w_final
    
    return predict
# EVOLVE-BLOCK-END
