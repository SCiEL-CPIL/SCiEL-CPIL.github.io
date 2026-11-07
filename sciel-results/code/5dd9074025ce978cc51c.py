# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import make_interp_spline, PchipInterpolator
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

_TOTAL_DEADLINE = time.time() + 325.0  # safety margin under 330s


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _fit_spline_1d(x, y, time_budget=25.0):
    """
    Fit a smooth spline to 1D noiseless data.
    Strategy: try multiple spline degrees and smoothing levels,
    pick the one with best validation performance.
    """
    n = len(x)
    if n < 5:
        # Not enough points for splines, fall back to polynomial
        return _fit_polynomial_1d(x, y, time_budget)
    
    # Sort by x
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    
    # Remove duplicates if any
    unique_mask = np.concatenate([True, np.diff(xs) > 1e-12])
    xs_u = xs[unique_mask]
    ys_u = ys[unique_mask]
    
    if len(xs_u) < 5:
        return _fit_polynomial_1d(x, y, time_budget)
    
    # Split into train/val for model selection
    n_val = max(10, n // 10)
    val_idx = np.random.choice(len(xs_u), n_val, replace=False)
    train_mask = np.ones(len(xs_u), dtype=bool)
    train_mask[val_idx] = False
    x_tr = xs_u[train_mask]
    y_tr = ys_u[train_mask]
    x_val = xs_u[val_idx]
    y_val = ys_u[val_idx]
    
    best_score = np.inf
    best_predictor = None
    start_time = time.time()
    
    # Try different spline configurations
    # Configuration: (k, s) where k is degree, s is smoothing factor
    # s=0 means exact interpolation
    configs = [
        (3, 0),      # Cubic, exact interpolation
        (3, 1e-10),  # Cubic, very light smoothing
        (3, 1e-8),   # Cubic, light smoothing
        (3, 1e-6),   # Cubic, moderate smoothing
        (3, 1e-4),   # Cubic, more smoothing
        (4, 0),      # Quartic, exact
        (4, 1e-10),  # Quartic, very light
        (5, 0),      # Quintic, exact
        (5, 1e-10),  # Quintic, very light
        (2, 0),      # Quadratic, exact
        (2, 1e-10),  # Quadratic, very light
    ]
    
    for k, s in configs:
        if time.time() - start_time > time_budget:
            break
        try:
            if s == 0:
                # Exact interpolation - need enough points
                if len(x_tr) < k + 1:
                    continue
                spl = make_interp_spline(x_tr, y_tr, k=k)
            else:
                spl = make_interp_spline(x_tr, y_tr, k=k, s=s)
            
            # Evaluate on validation set
            y_val_pred = spl(x_val)
            if not np.all(np.isfinite(y_val_pred)):
                continue
            mse = np.mean((y_val_pred - y_val) ** 2)
            
            if mse < best_score:
                best_score = mse
                # Refit on all data with same config
                if s == 0:
                    if len(xs_u) < k + 1:
                        continue
                    best_spl = make_interp_spline(xs_u, ys_u, k=k)
                else:
                    best_spl = make_interp_spline(xs_u, ys_u, k=k, s=s)
                best_predictor = best_spl
        except Exception:
            continue
    
    # If no spline worked, try PCHIP
    if best_predictor is None:
        try:
            pchip = PchipInterpolator(xs_u, ys_u)
            y_val_pred = pchip(x_val)
            if np.all(np.isfinite(y_val_pred)):
                best_predictor = pchip
        except Exception:
            pass
    
    if best_predictor is not None:
        spl = best_predictor
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            x_query = X[:, 0]
            # Clip to range of training data to avoid extrapolation issues
            x_min = xs_u[0]
            x_max = xs_u[-1]
            x_clip = np.clip(x_query, x_min, x_max)
            try:
                y_pred = spl(x_clip)
            except Exception:
                y_pred = np.full(len(x_clip), np.mean(ys_u))
            y_pred = np.nan_to_num(y_pred, nan=np.mean(ys_u), posinf=1e9, neginf=-1e9)
            return y_pred
        return predict
    else:
        # Fallback to polynomial
        return _fit_polynomial_1d(x, y, time_budget)


def _fit_polynomial_1d(x, y, time_budget=10.0):
    """Fit a polynomial to 1D data with degree selection."""
    n = len(x)
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    
    # Remove duplicates
    unique_mask = np.concatenate([True, np.diff(xs) > 1e-12])
    xs_u = xs[unique_mask]
    ys_u = ys[unique_mask]
    
    if len(xs_u) < 5:
        # Very few points, just use mean
        mean_y = np.mean(ys_u)
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(len(X), mean_y)
        return predict
    
    # Try different polynomial degrees
    best_score = np.inf
    best_degree = 2
    
    n_val = max(10, n // 10)
    val_idx = np.random.choice(len(xs_u), n_val, replace=False)
    train_mask = np.ones(len(xs_u), dtype=bool)
    train_mask[val_idx] = False
    x_tr = xs_u[train_mask]
    y_tr = ys_u[train_mask]
    x_val = xs_u[val_idx]
    y_val = ys_u[val_idx]
    
    max_deg = min(15, len(x_tr) - 1)
    for deg in range(1, max_deg + 1):
        if time.time() - time_budget > 10:
            break
        try:
            coeffs = np.polyfit(x_tr, y_tr, deg)
            y_val_pred = np.polyval(coeffs, x_val)
            if not np.all(np.isfinite(y_val_pred)):
                continue
            mse = np.mean((y_val_pred - y_val) ** 2)
            if mse < best_score:
                best_score = mse
                best_degree = deg
        except Exception:
            continue
    
    # Fit final polynomial on all data
    try:
        coeffs = np.polyfit(xs_u, ys_u, best_degree)
    except Exception:
        coeffs = np.polyfit(xs_u, ys_u, min(3, len(xs_u) - 1))
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_query = X[:, 0]
        y_pred = np.polyval(coeffs, x_query)
        y_pred = np.nan_to_num(y_pred, nan=np.mean(ys_u), posinf=1e9, neginf=-1e9)
        return y_pred
    return predict


def _fit_rbf_1d(x, y, time_budget=20.0):
    """Fit an RBF interpolant to 1D data."""
    n = len(x)
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    
    # Remove duplicates
    unique_mask = np.concatenate([True, np.diff(xs) > 1e-12])
    xs_u = xs[unique_mask]
    ys_u = ys[unique_mask]
    
    if len(xs_u) < 5:
        return _fit_polynomial_1d(x, y, time_budget)
    
    # Try different RBF widths
    best_score = np.inf
    best_width = None
    
    n_val = max(10, n // 10)
    val_idx = np.random.choice(len(xs_u), n_val, replace=False)
    train_mask = np.ones(len(xs_u), dtype=bool)
    train_mask[val_idx] = False
    x_tr = xs_u[train_mask]
    y_tr = ys_u[train_mask]
    x_val = xs_u[val_idx]
    y_val = ys_u[val_idx]
    
    # Estimate typical scale
    scale = np.std(xs_u) if len(xs_u) > 1 else 1.0
    
    widths = [0.1 * scale, 0.2 * scale, 0.5 * scale, 1.0 * scale, 2.0 * scale, 5.0 * scale]
    
    for width in widths:
        if time.time() - time_budget > 10:
            break
        try:
            # Compute RBF kernel matrix
            diff = x_tr[:, None] - x_tr[None, :]
            K = np.exp(-0.5 * (diff / width) ** 2)
            # Add small regularization for stability
            K_reg = K + 1e-8 * np.eye(len(x_tr))
            
            try:
                alpha = np.linalg.solve(K_reg, y_tr)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K_reg, y_tr, rcond=None)[0]
            
            # Evaluate on validation
            diff_val = x_val[:, None] - x_tr[None, :]
            K_val = np.exp(-0.5 * (diff_val / width) ** 2)
            y_val_pred = K_val @ alpha
            
            if not np.all(np.isfinite(y_val_pred)):
                continue
            mse = np.mean((y_val_pred - y_val) ** 2)
            
            if mse < best_score:
                best_score = mse
                best_width = width
        except Exception:
            continue
    
    if best_width is None:
        best_width = 0.5 * scale
    
    # Fit final RBF on all data
    try:
        diff = xs_u[:, None] - xs_u[None, :]
        K = np.exp(-0.5 * (diff / best_width) ** 2)
        K_reg = K + 1e-8 * np.eye(len(xs_u))
        alpha = np.linalg.solve(K_reg, ys_u)
    except Exception:
        return _fit_polynomial_1d(x, y, time_budget)
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_query = X[:, 0]
        diff = x_query[:, None] - xs_u[None, :]
        K = np.exp(-0.5 * (diff / best_width) ** 2)
        y_pred = K @ alpha
        y_pred = np.nan_to_num(y_pred, nan=np.mean(ys_u), posinf=1e9, neginf=-1e9)
        return y_pred
    return predict


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(1, -1)
    
    d = X_train.shape[1]
    
    if d != 1:
        # Fallback for multi-dimensional (shouldn't happen based on task)
        # Use polynomial features
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = features(X) @ w
            return np.nan_to_num(y, nan=np.mean(y_train), posinf=1e9, neginf=-1e9)
        return predict
    
    # 1D case: use spline-based approach
    x = X_train[:, 0]
    y = y_train
    
    # Check remaining time
    rem = _remaining()
    if rem < 10:
        # Not enough time, just use a simple polynomial
        return _fit_polynomial_1d(x, y, time_budget=5.0)
    
    # Try spline first (best for smooth 1D functions)
    try:
        predict_fn = _fit_spline_1d(x, y, time_budget=min(25.0, rem * 0.4))
        
        # Validate the predictor
        X_test_sample = X_train
        y_pred_sample = predict_fn(X_test_sample)
        if not np.all(np.isfinite(y_pred_sample)):
            raise ValueError("Non-finite predictions")
        
        # Check training error
        train_mse = np.mean((y_pred_sample - y) ** 2)
        y_var = np.var(y)
        if y_var > 1e-15:
            train_nrmse = train_mse / y_var
        else:
            train_nrmse = train_mse
        
        # If training error is very high, try RBF as backup
        if train_nrmse > 0.1 and _remaining() > 15:
            try:
                predict_fn_rbf = _fit_rbf_1d(x, y, time_budget=min(20.0, _remaining() * 0.5))
                y_pred_rbf = predict_fn_rbf(X_test_sample)
                if np.all(np.isfinite(y_pred_rbf)):
                    train_mse_rbf = np.mean((y_pred_rbf - y) ** 2)
                    train_nrmse_rbf = train_mse_rbf / y_var if y_var > 1e-15 else train_mse_rbf
                    if train_nrmse_rbf < train_nrmse:
                        predict_fn = predict_fn_rbf
            except Exception:
                pass
        
        return predict_fn
        
    except Exception:
        # Fallback to polynomial
        return _fit_polynomial_1d(x, y, time_budget=min(10.0, _remaining() * 0.5))
# EVOLVE-BLOCK-END
