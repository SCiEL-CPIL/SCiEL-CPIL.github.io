# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator, CubicSpline, make_interp_spline
from scipy.linalg import solve
from scipy.optimize import minimize_scalar

_TOTAL_DEADLINE = time.time() + 315.0


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(make_fn):
    """Wrap a prediction function to always return finite values."""
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict


def _fit_pchip(X, y):
    """Fit a PCHIP interpolator. Handles duplicates by averaging."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        # Handle duplicates: average y values for same x
        unique_xs, unique_indices, counts = np.unique(xs, return_inverse=True, return_counts=True)
        if len(unique_xs) < len(xs):
            ys_sum = np.zeros_like(unique_xs)
            np.add.at(ys_sum, unique_indices, ys)
            ys = ys_sum / counts
        
        if len(xs) < 4:
            return None

        s = PchipInterpolator(unique_xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_cubic_spline(X, y):
    """Fit a CubicSpline interpolator. Handles duplicates by averaging."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        # Handle duplicates: average y values for same x
        unique_xs, unique_indices, counts = np.unique(xs, return_inverse=True, return_counts=True)
        if len(unique_xs) < len(xs):
            ys_sum = np.zeros_like(unique_xs)
            np.add.at(ys_sum, unique_indices, ys)
            ys = ys_sum / counts
        
        if len(xs) < 4:
            return None

        s = CubicSpline(unique_xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_interp_spline(X, y, k=3):
    """Fit a general interpolation spline with configurable degree."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        unique_xs, unique_indices, counts = np.unique(xs, return_inverse=True, return_counts=True)
        if len(unique_xs) < len(xs):
            ys_sum = np.zeros_like(unique_xs)
            np.add.at(ys_sum, unique_indices, ys)
            ys = ys_sum / counts
        
        if len(xs) < k + 1:
            return None

        s = make_interp_spline(unique_xs, ys, k=k)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _legendre_features(X, degree):
    """Generate Legendre polynomial features (well-conditioned)."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        # Normalize to [-1, 1] for better conditioning
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-10:
            x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        else:
            x_norm = np.zeros_like(x)
        
        for p in range(degree + 1):
            if p == 0:
                cols.append(np.ones_like(x))
            elif p == 1:
                cols.append(x_norm)
            else:
                # Use recurrence relation for Legendre polynomials
                P0 = np.ones_like(x_norm)
                P1 = x_norm.copy()
                Pn = P1
                for n in range(1, p):
                    Pn = ((2*n + 1) * x_norm * P1 - n * P0) / (n + 1)
                    P0, P1 = P1, Pn
                cols.append(Pn)
    return np.column_stack(cols)


def _trig_features(X, max_k=12):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, max_k + 1):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
    return np.column_stack(cols)


def _exp_log_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        safe_x = np.clip(x, -20, 20)
        cols.append(np.exp(safe_x))
        cols.append(np.exp(-safe_x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1e-6))
        cols.append(np.exp(-x**2))
        cols.append(x * np.exp(-x**2))
        cols.append(np.abs(x))
        cols.append(np.sign(x) * np.log1p(np.abs(x)))
    return np.column_stack(cols)


def _mixed_features(X):
    """Combine well-conditioned features from multiple families."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-10:
            x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        else:
            x_norm = np.zeros_like(x)
        
        # Polynomial (normalized)
        for p in range(1, 10):
            cols.append(x_norm ** p)
        
        # Trigonometric
        for k in range(1, 10):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        
        # Special functions
        cols.append(np.tanh(x))
        cols.append(np.exp(-x ** 2))
        cols.append(np.exp(np.clip(x, -20, 20)))
        cols.append(np.exp(-np.clip(x, -20, 20)))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1e-6))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.abs(x))
    return np.column_stack(cols)


def _fit_robust_linear(F, y, ridge_alpha=1e-6):
    """Solve least squares with ridge regularization using SVD for stability."""
    try:
        n, p = F.shape
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        s_reg = s / (s**2 + ridge_alpha)
        Ut_y = U.T @ y
        w = Vt.T @ (Ut_y * s_reg)
        
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _try_fit_family(feat_fn, X, y, ridge_alpha=1e-6):
    """Try fitting a linear model on given features."""
    try:
        F = feat_fn(X)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        
        w = _fit_robust_linear(F, y, ridge_alpha)
        if w is None:
            return None, np.inf
        
        pred = F @ w
        if not np.all(np.isfinite(pred)):
            return None, np.inf
        
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        predict = _safe_predict(lambda Xp: feat_fn(Xp) @ w)
        return predict, rmse
    except Exception:
        return None, np.inf


def _fit_kernel_ridge(X, y, length_scale=1.0, alpha=1e-4):
    """Kernel Ridge Regression with RBF kernel."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
        K = np.exp(-0.5 * sq_dist) + alpha * np.eye(n)
        
        try:
            w = solve(K, y)
        except Exception:
            w, *_ = np.linalg.lstsq(K, y, rcond=None)
            
        if not np.all(np.isfinite(w)):
            return None

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            diff = Xpn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
            Ks = np.exp(-0.5 * sq_dist)
            preds = Ks @ w
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _estimate_smoothness(X, y):
    """Estimate the smoothness of the function using finite differences."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        if len(xs) < 10:
            return 1.0
        
        # Compute first and second differences
        dx = np.diff(xs)
        dy1 = np.diff(ys)
        dy2 = np.diff(dy1)
        
        # Filter out very small dx to avoid division issues
        mask = dx > 1e-8
        if mask.sum() < 5:
            return 1.0
        
        dx_f = dx[mask]
        dy1_f = dy1[mask]
        dy2_f = dy2[mask[:-1]] if mask.sum() > 1 else np.array([0.0])
        
        # Estimate derivatives
        d1 = dy1_f / dx_f
        d2 = dy2_f / (dx_f[:-1] * dx_f[1:]) if len(dx_f) > 1 else np.array([0.0])
        
        # Smoothness indicator: ratio of second derivative magnitude to first derivative
        d1_mag = np.abs(d1)
        d2_mag = np.abs(d2)
        
        if d1_mag.max() < 1e-10:
            return 0.5  # Nearly constant
        
        smoothness = np.median(d2_mag) / (np.median(d1_mag) + 1e-10)
        return float(np.clip(smoothness, 0.1, 10.0))
    except Exception:
        return 1.0


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_rmse = np.sqrt(np.mean((y_train - const) ** 2))
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # Estimate smoothness to guide model selection
    smoothness = _estimate_smoothness(X_train, y_train)

    # 1. Try interpolation methods (excellent for noiseless 1D data)
    if d == 1 and _remaining() > 5.0:
        pchip_pred = _fit_pchip(X_train, y_train)
        if pchip_pred is not None:
            pred = pchip_pred(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = pchip_pred

    if d == 1 and _remaining() > 5.0:
        cs_pred = _fit_cubic_spline(X_train, y_train)
        if cs_pred is not None:
            pred = cs_pred(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = cs_pred

    if d == 1 and _remaining() > 5.0:
        for k in [3, 4, 5]:
            if _remaining() < 3.0:
                break
            is_pred = _fit_interp_spline(X_train, y_train, k=k)
            if is_pred is not None:
                pred = is_pred(X_train)
                rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_predict = is_pred

    # 2. Try Kernel Ridge Regression
    if _remaining() > 10.0:
        for ls in [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
            if _remaining() < 5.0:
                break
            kr_pred = _fit_kernel_ridge(X_train, y_train, length_scale=ls, alpha=1e-4)
            if kr_pred is not None:
                pred = kr_pred(X_train)
                rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_predict = kr_pred

    # 3. Try Legendre polynomial regression (well-conditioned)
    for deg in [2, 4, 6, 8, 10, 12, 15, 20]:
        if _remaining() < 5.0:
            break
        predict, rmse = _try_fit_family(lambda Xp, d=deg: _legendre_features(Xp, d), X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # 4. Try Trigonometric Basis
    if _remaining() > 5.0:
        predict, rmse = _try_fit_family(lambda Xp: _trig_features(Xp, max_k=12), X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # 5. Try Exponential/Log Basis
    if _remaining() > 5.0:
        predict, rmse = _try_fit_family(_exp_log_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # 6. Try Mixed Basis
    if _remaining() > 5.0:
        predict, rmse = _try_fit_family(_mixed_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # 7. Try adaptive degree based on smoothness
    if _remaining() > 5.0:
        # If function is very smooth, lower degree may suffice
        # If rough, higher degree needed
        if smoothness < 0.5:
            max_deg = 8
        elif smoothness < 2.0:
            max_deg = 15
        else:
            max_deg = 25
        
        for deg in [3, 5, 7, 10, 15, 20, 25]:
            if deg > max_deg or _remaining() < 3.0:
                break
            predict, rmse = _try_fit_family(lambda Xp, d=deg: _legendre_features(Xp, d), X_train, y_train)
            if predict is not None and rmse < best_rmse:
                best_rmse = rmse
                best_predict = predict

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
