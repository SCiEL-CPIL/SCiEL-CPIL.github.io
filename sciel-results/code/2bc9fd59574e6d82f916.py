"""Symbolic regression via rich basis + ridge regression with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_TIME_LIMIT = 300.0
_SAFETY = 15.0
_START = time.time()


def _time_left():
    return _TIME_LIMIT - _SAFETY - (time.time() - _START)


def _build_rich_features(x):
    """x: 1D array. Returns a 2D array of basis functions (n, p)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n == 0:
        return np.empty((0, 0))

    cols = []
    # Polynomial terms
    for k in range(1, 10):
        cols.append(x ** k)
    # Trigonometric
    for mult in [0.5, 1.0, 2.0, 3.0, 4.0]:
        cols.append(np.sin(mult * x))
        cols.append(np.cos(mult * x))
    # Exponential (clipped for safety)
    x_safe = np.clip(x, -20, 20)
    for rate in [0.5, 1.0, 2.0]:
        cols.append(np.exp(rate * x_safe))
        cols.append(np.exp(-rate * x_safe))
    # Hyperbolic
    cols.append(np.tanh(x))
    cols.append(np.arctan(x))
    # Rational-ish / smooth
    cols.append(1.0 / (1.0 + x ** 2))
    cols.append(np.log1p(x ** 2))
    # Square root of abs
    cols.append(np.sqrt(np.abs(x)))
    # Absolute value
    cols.append(np.abs(x))
    # Mixed terms
    cols.append(x * np.sin(x))
    cols.append(x * np.cos(x))
    cols.append(x * np.exp(0.1 * x_safe))
    cols.append(np.sin(x) * np.cos(x))
    cols.append(x ** 2 * np.sin(x))
    cols.append(x ** 2 * np.cos(x))

    A = np.column_stack(cols)
    # Handle NaN/Inf
    A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)
    return A


def _scale_features(A):
    """Standardize each column to zero mean, unit variance (with guard)."""
    n, p = A.shape
    if p == 0:
        return A, np.zeros(0), np.ones(0)
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd[sd < 1e-12] = 1.0
    A_scaled = (A - mu) / sd
    return A_scaled, mu, sd


def _ridge_solve(A_scaled, y, lam):
    """Solve (A^T A + lam I) w = A^T y via Cholesky or lstsq fallback."""
    n, p = A_scaled.shape
    if p == 0:
        return np.zeros(0)
    try:
        G = A_scaled.T @ A_scaled + lam * np.eye(p)
        b = A_scaled.T @ y
        # Use Cholesky for speed and stability
        L = np.linalg.cholesky(G)
        w = np.linalg.solve(L.T, np.linalg.solve(L, b))
        return w
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        w, *_ = np.linalg.lstsq(A_scaled, y, rcond=None)
        return w


def _poly_fit(X, y, degree):
    """Polynomial fit of given degree."""
    x = X[:, 0]
    cols = [x ** k for k in range(degree + 1)]
    F = np.column_stack(cols) if degree > 0 else np.ones((len(X), 1))
    # Use ridge for stability
    lam = 1e-6
    try:
        w = np.linalg.solve(F.T @ F + lam * np.eye(len(w)), F.T @ y)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
    return lambda X: (np.column_stack([X[:, 0] ** k for k in range(degree + 1)]) if degree > 0 else np.ones((len(X), 1))) @ w


def _fourier_fit(X, y, n_terms):
    """Fourier series fit."""
    x = X[:, 0]
    cols = [np.ones_like(x)]
    for k in range(1, n_terms + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    F = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    
    def predict(X_new):
        x_new = X_new[:, 0]
        cols = [np.ones_like(x_new)]
        for k in range(1, n_terms + 1):
            cols.append(np.sin(k * x_new))
            cols.append(np.cos(k * x_new))
        F_new = np.column_stack(cols)
        return F_new @ w
    return predict


def fit(X_train, y_train):
    """Fit multiple models and return the best one based on training MSE."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    n, d = X_train.shape

    # Extract the single input column (task says 1 input)
    if d >= 1:
        x_train = X_train[:, 0]
    else:
        x_train = X_train.ravel()

    candidates = []

    # 1. Rich basis + ridge regression
    if _time_left() > 10:
        try:
            A = _build_rich_features(x_train)
            A_scaled, mu, sd = _scale_features(A)
            
            best_w = None
            best_mse = np.inf
            lams = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
            
            for lam in lams:
                if _time_left() < 5.0:
                    break
                w = _ridge_solve(A_scaled, y_train, lam)
                pred = A_scaled @ w
                mse = np.mean((pred - y_train) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_w = w
            
            if best_w is not None:
                def rich_predict(X_new):
                    X_new = np.asarray(X_new, dtype=np.float64)
                    if X_new.ndim == 1:
                        X_new = X_new.reshape(-1, 1)
                    if X_new.shape[1] >= 1:
                        x = X_new[:, 0]
                    else:
                        x = X_new.ravel()
                    A_new = _build_rich_features(x)
                    A_new_scaled = (A_new - mu) / sd
                    return A_new_scaled @ best_w
                candidates.append((best_mse, rich_predict))
        except Exception:
            pass

    # 2. Polynomial fits of various degrees
    for deg in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
        if _time_left() < 5:
            break
        try:
            pred_fn = _poly_fit(X_train, y_train, deg)
            train_pred = pred_fn(X_train)
            mse = np.mean((train_pred - y_train) ** 2)
            candidates.append((mse, pred_fn))
        except Exception:
            pass

    # 3. Fourier series
    for n_terms in [3, 5, 8, 12, 20, 30]:
        if _time_left() < 5:
            break
        try:
            pred_fn = _fourier_fit(X_train, y_train, n_terms)
            train_pred = pred_fn(X_train)
            mse = np.mean((train_pred - y_train) ** 2)
            candidates.append((mse, pred_fn))
        except Exception:
            pass

    # 4. Spline interpolation
    if _time_left() > 5:
        try:
            from scipy.interpolate import interp1d
            x = X_train[:, 0]
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y_train[order]
            if len(x_sorted) >= 4:
                f = interp1d(x_sorted, y_sorted, kind='cubic', fill_value='extrapolate')
            else:
                f = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
            pred_fn = lambda X: f(X[:, 0])
            train_pred = pred_fn(X_train)
            mse = np.mean((train_pred - y_train) ** 2)
            candidates.append((mse, pred_fn))
        except Exception:
            pass

    # Select best model by training MSE
    if not candidates:
        # Fallback: linear fit
        pred_fn = _poly_fit(X_train, y_train, 1)
        return pred_fn

    candidates.sort(key=lambda t: t[0])
    best_mse, best_pred = candidates[0]

    # Final safety check
    def safe_predict(X_new):
        try:
            result = best_pred(X_new)
            result = np.asarray(result, dtype=np.float64).flatten()
            # Ensure finite values
            if not np.all(np.isfinite(result)):
                result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
        except Exception:
            return np.zeros(len(X_new))

    return safe_predict
# EVOLVE-BLOCK-END
