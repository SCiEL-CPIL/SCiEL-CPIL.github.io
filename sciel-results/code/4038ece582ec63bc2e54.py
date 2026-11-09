"""Symbolic regression via robust piecewise polynomial interpolation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d

# Explicit deadline guard: total wall-clock budget
_TOTAL_DEADLINE = 325.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def fit(X_train, y_train):
    """
    Fit a regression model to 1D noiseless data.
    Strategy:
      1. Try PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.
         This is robust and avoids oscillations.
      2. If that fails, try CubicSpline with not-a-knot boundary conditions.
      3. If that fails, try a global polynomial fit.
      4. Final fallback: linear interpolation.
    """
    X = np.asarray(X_train, dtype=np.float64).reshape(-1)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n = len(X)
    
    if n < 2:
        # Too few points; use constant prediction
        mean_y = np.mean(y) if n > 0 else 0.0
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
    
    if len(Xs_u) < 2:
        mean_y = np.mean(ys_u)
        return lambda X_test: np.full(len(np.asarray(X_test).reshape(-1)), mean_y)

    x_min, x_max = Xs_u[0], Xs_u[-1]
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        y_var = 1.0

    # Helper to check if a model is good
    def _check_model(model, Xs_u, ys_u, name):
        try:
            pred = model(Xs_u)
            if not np.all(np.isfinite(pred)):
                return False
            mse = np.mean((pred - ys_u) ** 2)
            # For noiseless data, MSE should be extremely small
            if mse < 1e-8 * y_var:
                return True
            else:
                # If MSE is high, the model might be wrong or data is not 1D smooth
                # But for symbolic regression, it should fit well.
                # Let's allow a slightly higher tolerance for spline wiggles
                if mse < 1e-4 * y_var:
                    return True
        except Exception:
            return False
        return False

    # 1. Try PCHIP
    pchip_model = None
    try:
        if len(Xs_u) >= 2:
            pchip_model = PchipInterpolator(Xs_u, ys_u, extrapolate=True)
            if _check_model(pchip_model, Xs_u, ys_u, "PCHIP"):
                def predict_pchip(X_test):
                    Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
                    pred = pchip_model(Xt)
                    pred = np.where(np.isfinite(pred), pred, y_mean)
                    return pred
                return predict_pchip
    except Exception:
        pchip_model = None

    # 2. Try CubicSpline
    spline_model = None
    try:
        if len(Xs_u) >= 4:
            # Try not-a-knot first, then natural
            for bc in ['not-a-knot', 'natural']:
                try:
                    cs = CubicSpline(Xs_u, ys_u, bc_type=bc, extrapolate=True)
                    if _check_model(cs, Xs_u, ys_u, f"CubicSpline-{bc}"):
                        spline_model = cs
                        break
                except Exception:
                    continue
            if spline_model:
                def predict_spline(X_test):
                    Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
                    pred = spline_model(Xt)
                    pred = np.where(np.isfinite(pred), pred, y_mean)
                    return pred
                return predict_spline
    except Exception:
        spline_model = None

    # 3. Try Global Polynomial Fit
    # Scale X to [-1, 1] for stability
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
    center = (x_min + x_max) / 2
    
    # Determine max degree based on number of points
    max_deg = min(15, len(Xs_u) - 1)
    poly_coeffs = None
    for deg in range(10, 0, -1):
        if deg > len(Xs_u) - 1:
            continue
        try:
            Xs_scaled = (Xs_u - center) / (x_range / 2)
            # Use polyfit with a warning filter
            coeffs = np.polyfit(Xs_scaled, ys_u, deg)
            
            def predict_poly(X_test, coeffs=coeffs, deg=deg):
                Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
                Xt_scaled = (Xt - center) / (x_range / 2)
                pred = np.polyval(coeffs, Xt_scaled)
                pred = np.where(np.isfinite(pred), pred, y_mean)
                return pred
            
            # Check training error
            pred_train = predict_poly(Xs_u)
            mse = np.mean((pred_train - ys_u) ** 2)
            if mse < 1e-8 * y_var:
                return predict_poly
        except Exception:
            continue

    # 4. Fallback: Linear Interpolation with Extrapolation
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

    # 5. Final Fallback: Nearest Neighbor
    def predict_nn(X_test):
        Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
        # Find nearest training point for each test point
        # Use searchsorted for efficiency
        idx = np.searchsorted(Xs_u, Xt)
        # Clip indices to valid range
        idx = np.clip(idx, 1, len(Xs_u) - 1)
        
        # Compare distance to left and right neighbor
        left_idx = idx - 1
        right_idx = idx
        
        d_left = np.abs(Xt - Xs_u[left_idx])
        d_right = np.abs(Xt - Xs_u[right_idx])
        
        # If Xt is exactly at a knot, searchsorted returns the index of that knot (or next if not found)
        # If Xt < Xs_u[0], idx=0, left_idx=-1 (bad). Handle boundaries.
        
        # Simpler robust NN:
        idx_nn = np.argmin(np.abs(Xs_u[:, None] - Xt[None, :]), axis=0)
        pred = ys_u[idx_nn]
        pred = np.where(np.isfinite(pred), pred, y_mean)
        return pred
    
    return predict_nn
# EVOLVE-BLOCK-END
