"""Symbolic regression over 10 hidden equations: comprehensive feature library + ridge selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import lstsq

_START = time.time()
_DEADLINE = 300.0  # safety margin vs 330s kill

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _safe_div(a, b, eps=1e-8):
    """Element-wise division with safe denominator."""
    return a / np.where(np.abs(b) < eps, np.sign(b) * eps + (b == 0) * eps, b)

def _build_features(X):
    """Build a comprehensive feature matrix from input X."""
    n, d = X.shape
    cols = [np.ones(n)]
    
    # Linear terms
    for j in range(d):
        cols.append(X[:, j])
    
    # Squares
    for j in range(d):
        cols.append(X[:, j] ** 2)
    
    # Cubes
    for j in range(d):
        cols.append(X[:, j] ** 3)
    
    # Pairwise products
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
    
    # Pairwise sums and differences
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] + X[:, j])
            cols.append(X[:, i] - X[:, j])
    
    # Trigonometric functions of individual inputs
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
        cols.append(np.sin(2 * X[:, j]))
        cols.append(np.cos(2 * X[:, j]))
    
    # Trigonometric functions of pairwise sums/differences
    for i in range(d):
        for j in range(i + 1, d):
            s = X[:, i] + X[:, j]
            diff = X[:, i] - X[:, j]
            cols.append(np.sin(s))
            cols.append(np.cos(s))
            cols.append(np.sin(diff))
            cols.append(np.cos(diff))
    
    # Exponential functions (clipped to avoid overflow)
    for j in range(d):
        x_clip = np.clip(X[:, j], -50, 50)
        cols.append(np.exp(x_clip))
        cols.append(np.exp(-x_clip))
    
    # Sigmoid / tanh
    for j in range(d):
        cols.append(np.tanh(X[:, j]))
        x_clip = np.clip(X[:, j], -500, 500)
        cols.append(1.0 / (1.0 + np.exp(-x_clip)))
    
    # Rational functions (safe)
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(_safe_div(X[:, i], X[:, j]))
            cols.append(_safe_div(X[:, j], X[:, i]))
            cols.append(_safe_div(1.0, X[:, i]))
            cols.append(_safe_div(1.0, X[:, j]))
    
    # Product of all inputs
    if d >= 1:
        p = np.ones(n)
        for j in range(d):
            p = p * X[:, j]
        cols.append(p)
    
    # Sum of all inputs
    s = np.sum(X, axis=1)
    cols.append(s)
    cols.append(s ** 2)
    cols.append(np.sin(s))
    cols.append(np.cos(s))
    cols.append(np.tanh(s))
    
    # Norm
    norm = np.sqrt(np.sum(X ** 2, axis=1))
    cols.append(norm)
    cols.append(np.sin(norm))
    cols.append(np.cos(norm))
    
    # Stack and remove non-finite columns
    A = np.column_stack(cols)
    # Replace non-finite values with 0 and track which columns are valid
    mask = np.isfinite(A)
    A_clean = np.where(mask, A, 0.0)
    col_valid = mask.all(axis=0)
    A_clean = A_clean[:, col_valid]
    
    return A_clean

def _ridge_fit(A, y, alpha):
    """Solve ridge regression: (A^T A + alpha*I) w = A^T y"""
    n, p = A.shape
    try:
        AtA = A.T @ A
        Aty = A.T @ y
        # Add small ridge for numerical stability
        reg = (alpha + 1e-10) * np.eye(p)
        w = np.linalg.solve(AtA + reg, Aty)
        pred = A @ w
        if not np.all(np.isfinite(pred)):
            return None
        err = np.sqrt(np.mean((pred - y) ** 2))
        return w, err
    except Exception:
        return None

def _select_alpha(A, y, alphas=None):
    """Try multiple ridge alphas and pick the best."""
    if alphas is None:
        alphas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    best = None
    for alpha in alphas:
        res = _ridge_fit(A, y, alpha)
        if res is not None:
            w, err = res
            if best is None or err < best[0]:
                best = (err, w, alpha)
    
    return best

def _make_predict(A_train, w, X_train):
    """Create a predict function using precomputed feature transform."""
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        A = _build_features(X)
        # Ensure feature dimensions match
        if A.shape[1] != len(w):
            # Fallback: just use linear
            return X @ w[:X.shape[1]] + w[X.shape[1]] if len(w) > X.shape[1] else X @ w
        return A @ w
    return predict

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)
    
    if _remaining() < 10.0:
        # Emergency fallback: simple linear
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.column_stack([np.ones(len(X)), X]) @ w
        return predict
    
    # Build comprehensive feature set
    A = _build_features(X)
    
    if A.shape[1] == 0:
        # Fallback
        w = np.zeros(1)
        def predict(X):
            return np.ones(len(X)) * w[0]
        return predict
    
    # Try ridge with various alphas
    best = _select_alpha(A, y)
    
    if best is None:
        # Fallback to OLS
        try:
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ w
            if not np.all(np.isfinite(pred)):
                w = np.zeros(A.shape[1])
            err = np.sqrt(np.mean((pred - y) ** 2)) if np.all(np.isfinite(pred)) else float('inf')
        except Exception:
            w = np.zeros(A.shape[1])
            err = float('inf')
    else:
        err, w, alpha = best
    
    return _make_predict(A, w, X)
# EVOLVE-BLOCK-END
