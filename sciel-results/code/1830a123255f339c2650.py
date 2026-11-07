# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import make_interp_spline, CubicSpline, PchipInterpolator
from scipy.linalg import solve

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


def _legendre_features(X, degree):
    """
    Compute Legendre polynomial features scaled to [-1, 1].
    This is numerically stable compared to raw powers.
    """
    x = X[:, 0]
    # Scale to [-1, 1]
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-9:
        x_norm = np.zeros_like(x)
    else:
        x_norm = 2.0 * (x - x_min) / x_range - 1.0
    
    cols = [np.ones(X.shape[0])]
    # P_0 = 1, P_1 = x
    # Recurrence: (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
    if degree >= 1:
        cols.append(x_norm.copy())
    if degree >= 2:
        P_nm1 = np.ones_like(x)
        P_n = x_norm.copy()
        for k in range(2, degree + 1):
            P_np1 = ((2 * k - 1) * x_norm * P_n - (k - 1) * P_nm1) / k
            cols.append(P_np1)
            P_nm1, P_n = P_n, P_np1
            
    return np.column_stack(cols)


def _trig_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, 8):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
    return np.column_stack(cols)


def _exp_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        # Safe log
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1e-6))
        cols.append(x * np.exp(-x))
        cols.append(x * np.exp(x))
        cols.append(np.exp(-x**2))
    return np.column_stack(cols)


def _fit_ols(F, y, ridge_alpha=1e-6):
    """Solve least squares with ridge regularization."""
    try:
        n, p = F.shape
        # Ridge: (F^T F + alpha I) w = F^T y
        # We add alpha to all coefficients except intercept? 
        # Standard ridge adds to all. Let's add to all for stability.
        A = F.T @ F + ridge_alpha * np.eye(p)
        b = F.T @ y
        w = solve(A, b)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _try_fit_family(feat_fn, X, y, ridge_alpha=1e-6):
    """Try fitting a linear model on given features. Returns (predict_fn, rmse) or (None, inf)."""
    try:
        F = feat_fn(X)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        w = _fit_ols(F, y, ridge_alpha)
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


def _fit_spline_interp(X, y, num_knots=20, method='pchip'):
    """Fit a spline via interpolation on sorted points."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 10:
            return None
        num_knots = min(num_knots, n)
        if num_knots < 5:
            num_knots = 5
        knot_idx = np.linspace(0, n - 1, num_knots, dtype=int)
        knot_x = xs[knot_idx]
        knot_y = ys[knot_idx]
        
        if method == 'pchip':
            s = PchipInterpolator(knot_x, knot_y)
        else:
            s = make_interp_spline(knot_x, knot_y, k=3)
        
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


def _fit_kernel_ridge(X, y, length_scale=1.0, alpha=1e-6):
    """Kernel Ridge Regression with RBF kernel."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
        K = np.exp(-0.5 * sq_dist) + alpha * np.eye(n)
        
        w = solve(K, y)
        
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

    # 1. Try Spline Interpolation (often best for noiseless 1D)
    if d == 1:
        for nk in [15, 25, 40]:
            if _remaining() < 5.0:
                break
            for meth in ['pchip', 'cubic']:
                sp_predict = _fit_spline_interp(X_train, y_train, num_knots=nk, method=meth)
                if sp_predict is not None:
                    pred = sp_predict(X_train)
                    rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_predict = sp_predict

    # 2. Try Legendre Polynomials (Stable)
    if _remaining() > 10.0:
        for deg in [3, 5, 8, 12, 16]:
            if _remaining() < 5.0:
                break
            predict, rmse = _try_fit_family(lambda Xp, d=deg: _legendre_features(Xp, d), X_train, y_train)
            if predict is not None and rmse < best_rmse:
                best_rmse = rmse
                best_predict = predict

    # 3. Try Trig Features
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_trig_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # 4. Try Exp/Log Features
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_exp_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # 5. Try Kernel Ridge Regression
    if _remaining() > 15.0:
        for ls in [0.5, 1.0, 2.0, 5.0]:
            if _remaining() < 5.0:
                break
            kr_predict = _fit_kernel_ridge(X_train, y_train, length_scale=ls)
            if kr_predict is not None:
                pred = kr_predict(X_train)
                rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_predict = kr_predict

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
