"""Symbolic regression with 1D Gaussian Process (Matérn-5/2) and feature-based models."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 330.0
SAFETY_MARGIN = 30.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _matern52_kernel(x1, x2, length_scale, signal_var):
    """Matérn-5/2 kernel for 1D inputs."""
    d = np.abs(x1 - x2)
    sqrt3_d = np.sqrt(3.0) * d / length_scale
    return signal_var * (1 + sqrt3_d + sqrt3_d ** 2 / 3.0) * np.exp(-sqrt3_d)


def _fit_gp(X_train, y_train, length_scale, signal_var, noise_var):
    """Fit a Gaussian Process with Matérn-5/2 kernel."""
    x = X_train[:, 0]
    n = len(x)
    
    K = _matern52_kernel(x, x, length_scale, signal_var)
    K += noise_var * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        # Add jitter and retry
        K += 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return None, float('inf')
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    def predict(X):
        x_test = X[:, 0]
        K_s = _matern52_kernel(x_test, x, length_scale, signal_var)
        return K_s @ alpha
    
    # Training MSE
    y_pred = K @ alpha - noise_var * alpha
    mse = np.mean((y_pred - y_train) ** 2)
    return predict, mse


def _gp_cv(X_train, y_train, length_scale, signal_var, noise_var, n_folds=5):
    """5-fold CV for GP."""
    n = len(X_train)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    cv_errors = []
    
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        predict, _ = _fit_gp(X_tr, y_tr, length_scale, signal_var, noise_var)
        if predict is None:
            cv_errors.append(float('inf'))
        else:
            y_pred = predict(X_val)
            cv_errors.append(np.mean((y_pred - y_val) ** 2))
    
    return np.mean(cv_errors)


def _poly_features(X, degree):
    """Polynomial features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    return np.column_stack(cols)


def _chebyshev_features(X, n_terms):
    """Chebyshev polynomial features mapped to [-1, 1]."""
    x = X[:, 0]
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-10:
        x_norm = np.zeros_like(x)
    else:
        x_norm = (x - (x_min + x_max) / 2) / ((x_max - x_min) / 2)
    x_norm = np.clip(x_norm, -1.0, 1.0)
    
    cols = []
    for k in range(n_terms + 1):
        if k == 0:
            cols.append(np.ones(len(x)))
        elif k == 1:
            cols.append(x_norm.copy())
        else:
            T_prev = cols[-2]
            T_curr = cols[-1]
            T_next = 2 * x_norm * T_curr - T_prev
            cols.append(T_next)
    return np.column_stack(cols)


def _fourier_features(X, n_harmonics):
    """Fourier series features."""
    x = X[:, 0]
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-10:
        x_norm = np.zeros_like(x)
    else:
        x_norm = (x - (x_min + x_max) / 2) / ((x_max - x_min) / 2) * 2 * np.pi
    
    cols = [np.ones(len(x))]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * x_norm))
        cols.append(np.cos(k * x_norm))
    return np.column_stack(cols)


def _rich_features(X):
    """Rich feature set: poly, trig, exp, rational."""
    x = X[:, 0]
    n = len(x)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-10:
        x_norm = np.zeros_like(x)
    else:
        x_norm = (x - (x_min + x_max) / 2) / ((x_max - x_min) / 2)
    
    cols = [np.ones(n)]
    
    # Polynomial up to degree 8
    for k in range(1, 9):
        cols.append(x ** k)
    
    # Chebyshev up to degree 10
    for k in range(1, 11):
        if k == 1:
            cols.append(x_norm.copy())
        else:
            T_prev = cols[-2] if len(cols) >= 2 else np.ones(n)
            T_curr = cols[-1]
            T_next = 2 * x_norm * T_curr - T_prev
            cols.append(T_next)
    
    # Fourier
    for k in range(1, 7):
        cols.append(np.sin(k * x_norm * np.pi))
        cols.append(np.cos(k * x_norm * np.pi))
    
    # Exponential
    x_clip = np.clip(x, -5, 5)
    cols.append(np.exp(0.5 * x_clip))
    cols.append(np.exp(-0.5 * x_clip))
    cols.append(np.exp(1.0 * x_clip))
    cols.append(np.exp(-1.0 * x_clip))
    
    # Rational
    denom = 1 + x ** 2
    cols.append(x / denom)
    cols.append(x ** 2 / denom)
    cols.append(x ** 3 / denom)
    
    # Abs and sqrt
    cols.append(np.abs(x))
    pos_mask = x > 0
    if np.any(pos_mask):
        sqrt_x = np.zeros(n)
        sqrt_x[pos_mask] = np.sqrt(x[pos_mask])
        cols.append(sqrt_x)
    
    return np.column_stack(cols)


def _fit_ridge(Xf, y, alpha):
    """Fit ridge regression."""
    n, p = Xf.shape
    AtA = Xf.T @ Xf + alpha * np.eye(p)
    Aty = Xf.T @ y
    try:
        w = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    return w


def _cv_feature_model(features_func, X_train, y_train, alpha, n_folds=5):
    """CV for feature-based model."""
    n = len(X_train)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    cv_errors = []
    
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        Xf_tr = features_func(X_tr)
        Xf_val = features_func(X_val)
        
        w = _fit_ridge(Xf_tr, y_tr, alpha)
        y_pred = Xf_val @ w
        cv_errors.append(np.mean((y_pred - y_val) ** 2))
    
    return np.mean(cv_errors)


def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    d = X_train.shape[1]
    if d != 1:
        # Fallback for multi-dimensional
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w
    
    best_cv_mse = float('inf')
    best_predict = None
    
    # --- GP with grid search over hyperparameters ---
    x_range = X_train[:, 0]
    x_min, x_max = x_range.min(), x_range.max()
    x_span = max(x_max - x_min, 1e-6)
    
    # Grid of hyperparameters
    length_scales = [x_span * f for f in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]]
    signal_vars = [np.var(y_train) * f for f in [0.5, 1.0, 2.0, 5.0]]
    noise_vars = [1e-6, 1e-4, 1e-3, 1e-2]
    
    # First, do a coarse grid search with CV
    for ls in length_scales:
        if _time_left() < 20.0:
            break
        for sv in signal_vars:
            if _time_left() < 10.0:
                break
            for nv in noise_vars:
                if _time_left() < 5.0:
                    break
                try:
                    cv_mse = _gp_cv(X_train, y_train, ls, sv, nv)
                    if cv_mse < best_cv_mse:
                        best_cv_mse = cv_mse
                        predict, _ = _fit_gp(X_train, y_train, ls, sv, nv)
                        if predict is not None:
                            best_predict = predict
                except Exception:
                    continue
    
    # --- Feature-based models ---
    feature_candidates = [
        ('poly5', lambda X: _poly_features(X, 5), 1e-4),
        ('poly8', lambda X: _poly_features(X, 8), 1e-3),
        ('poly12', lambda X: _poly_features(X, 12), 1e-2),
        ('cheb10', lambda X: _chebyshev_features(X, 10), 1e-4),
        ('cheb15', lambda X: _chebyshev_features(X, 15), 1e-3),
        ('fourier8', lambda X: _fourier_features(X, 8), 1e-4),
        ('fourier12', lambda X: _fourier_features(X, 12), 1e-3),
        ('rich', _rich_features, 1e-3),
        ('rich', _rich_features, 1e-2),
        ('rich', _rich_features, 1e-1),
    ]
    
    for name, feat_func, alpha in feature_candidates:
        if _time_left() < 10.0:
            break
        try:
            cv_mse = _cv_feature_model(feat_func, X_train, y_train, alpha)
            if cv_mse < best_cv_mse:
                best_cv_mse = cv_mse
                Xf = feat_func(X_train)
                w = _fit_ridge(Xf, y_train, alpha)
                best_predict = lambda X, ff=feat_func, ww=w: ff(X) @ ww
        except Exception:
            continue
    
    # Fallback
    if best_predict is None:
        Xf = _poly_features(X_train, 5)
        w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
        best_predict = lambda X, ww=w: _poly_features(X, 5) @ ww
    
    return best_predict
# EVOLVE-BLOCK-END
