"""Symbolic regression via cubic spline interpolation with safety fallbacks."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline, interp1d

# Explicit deadline guard: total wall-clock budget
_TOTAL_DEADLINE = 330.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def fit(X_train, y_train):
    """
    Fit a regression model to 1D noiseless data.
    Strategy:
      1. Try cubic spline interpolation (exact for smooth functions).
      2. If that fails or produces non-finite values, fall back to polynomial fit.
      3. If still bad, use nearest-neighbor interpolation.
    """
    X = np.asarray(X_train, dtype=np.float64).reshape(-1)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n = len(X)
    
    if n < 4:
        # Too few points for spline; use constant prediction
        mean_y = np.mean(y)
        return lambda X_test: np.full(len(np.asarray(X_test).reshape(-1)), mean_y)
    
    # Sort by X
    sort_idx = np.argsort(X)
    Xs = X[sort_idx]
    ys = y[sort_idx]
    
    # Remove duplicate X values (keep last)
    unique_mask = np.ones(len(Xs), dtype=bool)
    if len(Xs) > 1:
        unique_mask[1:] = Xs[1:] != Xs[:-1]
    Xs_u = Xs[unique_mask]
    ys_u = ys[unique_mask]
    
    x_min, x_max = Xs_u[0], Xs_u[-1]
    y_mean = np.mean(y)
    
    # Check if data is reasonably smooth (not wildly oscillating)
    # Compute a rough smoothness score: mean absolute second difference
    if len(ys_u) >= 3:
        d2 = np.abs(np.diff(ys_u, 2))
        smoothness = np.mean(d2) / (np.std(ys_u) + 1e-12)
    else:
        smoothness = 0.0
    
    # Try cubic spline interpolation
    spline_pred = None
    try:
        if len(Xs_u) >= 4 and smoothness < 10.0:
            cs = CubicSpline(Xs_u, ys_u, bc_type='not-a-knot')
            # Test on training points to verify
            test_X = Xs_u
            test_pred = cs(test_X)
            if np.all(np.isfinite(test_pred)):
                train_mse = np.mean((test_pred - ys_u) ** 2)
                if train_mse < 1e-6 * (np.var(y) + 1e-12):
                    spline_pred = cs
    except Exception:
        spline_pred = None
    
    if spline_pred is not None:
        cs = spline_pred
        
        def predict(X_test):
            Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
            # Clip to training range to avoid extrapolation issues
            Xt_clipped = np.clip(Xt, x_min, x_max)
            pred = cs(Xt_clipped)
            # Replace any non-finite values with mean
            pred = np.where(np.isfinite(pred), pred, y_mean)
            return pred
        
        return predict
    
    # Fallback 1: Linear interpolation
    try:
        f_interp = interp1d(Xs_u, ys_u, kind='linear', fill_value='extrapolate', bounds_error=False)
        
        def predict_linear(X_test):
            Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
            pred = f_interp(Xt)
            pred = np.where(np.isfinite(pred), pred, y_mean)
            return pred
        
        return predict_linear
    except Exception:
        pass
    
    # Fallback 2: Polynomial fit (degree 4)
    try:
        # Scale X to [-1, 1] for numerical stability
        x_range = x_max - x_min
        if x_range < 1e-12:
            x_range = 1.0
        Xs_scaled = (Xs_u - (x_min + x_max) / 2) / (x_range / 2)
        
        deg = min(4, len(Xs_u) - 1)
        coeffs = np.polyfit(Xs_scaled, ys_u, deg)
        
        def predict_poly(X_test):
            Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
            Xt_scaled = (Xt - (x_min + x_max) / 2) / (x_range / 2)
            pred = np.polyval(coeffs, Xt_scaled)
            pred = np.where(np.isfinite(pred), pred, y_mean)
            return pred
        
        return predict_poly
    except Exception:
        pass
    
    # Final fallback: nearest neighbor
    def predict_nn(X_test):
        Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
        # Find nearest training point for each test point
        idx = np.argmin(np.abs(Xs_u[:, None] - Xt[None, :]), axis=0)
        pred = ys_u[idx]
        pred = np.where(np.isfinite(pred), pred, y_mean)
        return pred
    
    return predict_nn
# EVOLVE-BLOCK-END
