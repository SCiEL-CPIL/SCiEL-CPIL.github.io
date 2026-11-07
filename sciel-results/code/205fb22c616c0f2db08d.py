# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from numpy.polynomial import chebyshev as C

_TOTAL_DEADLINE = time.time() + 320.0


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


def _fit_chebyshev_collocation(X, y, max_degree=None):
    """
    Chebyshev collocation: for noiseless 1D data, interpolate exactly.
    If max_degree >= n, this is exact interpolation.
    """
    try:
        x = X[:, 0]
        n = len(x)
        if max_degree is None:
            max_degree = n - 1
        max_degree = min(max_degree, 120)  # cap for numerical stability
        
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-12:
            return None, np.inf
        
        x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        
        # Use Chebyshev nodes for evaluation, but we need to fit to our data
        # Build Vandermonde matrix
        F = C.chebvander(x_norm, max_degree)
        
        # Ridge solve for stability
        ridge = 1e-10
        A = F.T @ F + ridge * np.eye(max_degree + 1)
        b = F.T @ y
        
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(F, y, rcond=None)[0]
        
        pred = F @ w
        if not np.all(np.isfinite(pred)):
            return None, np.inf
        
        mse = np.mean((pred - y) ** 2)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xp_norm = 2.0 * (xp - x_min) / (x_max - x_min) - 1.0
            Fp = C.chebvander(xp_norm, max_degree)
            preds = Fp @ w
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        
        return _safe_predict(predict), mse
    except Exception:
        return None, np.inf


def _fit_rich_basis(X, y):
    """
    Fit using a rich basis of polynomial, trig, and exponential features.
    This can capture functions like sin(x), exp(x), x^2*sin(x), etc.
    """
    try:
        x = X[:, 0]
        n = len(x)
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min
        if x_range < 1e-12:
            return None, np.inf
        
        # Normalize x to [0, 1] for feature construction
        x_norm = (x - x_min) / x_range
        
        # Build feature matrix
        features = [np.ones(n)]
        
        # Polynomial features up to degree 10
        for k in range(1, 11):
            features.append(x_norm ** k)
        
        # Sine/cosine features with various frequencies
        for freq in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
            features.append(np.sin(freq * np.pi * x_norm))
            features.append(np.cos(freq * np.pi * x_norm))
        
        # Exponential features
        for base in [0.5, 1.0, 2.0]:
            features.append(np.exp(base * x_norm))
            features.append(np.exp(-base * x_norm))
        
        # Mixed features
        features.append(x_norm * np.sin(2.0 * np.pi * x_norm))
        features.append(x_norm * np.cos(2.0 * np.pi * x_norm))
        features.append(np.exp(x_norm) * np.sin(2.0 * np.pi * x_norm))
        
        F = np.column_stack(features)
        
        if not np.all(np.isfinite(F)):
            return None, np.inf
        
        # Ridge regression
        p = F.shape[1]
        ridge = 1e-6
        A = F.T @ F + ridge * np.eye(p)
        b = F.T @ y
        
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(F, y, rcond=None)[0]
        
        pred = F @ w
        if not np.all(np.isfinite(pred)):
            return None, np.inf
        
        mse = np.mean((pred - y) ** 2)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xn = (xp - x_min) / x_range
            feats = [np.ones(len(xn))]
            for k in range(1, 11):
                feats.append(xn ** k)
            for freq in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
                feats.append(np.sin(freq * np.pi * xn))
                feats.append(np.cos(freq * np.pi * xn))
            for base in [0.5, 1.0, 2.0]:
                feats.append(np.exp(base * xn))
                feats.append(np.exp(-base * xn))
            feats.append(xn * np.sin(2.0 * np.pi * xn))
            feats.append(xn * np.cos(2.0 * np.pi * xn))
            feats.append(np.exp(xn) * np.sin(2.0 * np.pi * xn))
            Fp = np.column_stack(feats)
            preds = Fp @ w
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        
        return _safe_predict(predict), mse
    except Exception:
        return None, np.inf


def _fit_pchip_interp(X, y):
    """
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) - monotone cubic.
    Exact for noiseless data, O(n) evaluation.
    """
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        # Remove duplicate x values
        unique_x, inverse, counts = np.unique(xs, return_inverse=True, return_counts=True)
        if len(unique_x) < len(xs):
            y_avg = np.zeros(len(unique_x))
            np.add.at(y_avg, inverse, ys)
            y_avg /= counts
            xs, ys = unique_x, y_avg
        
        if len(xs) < 2:
            return None, np.inf
        
        interp = PchipInterpolator(xs, ys)
        
        pred = interp(xs)
        mse = np.mean((pred - ys) ** 2)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            preds = interp(xp)
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        
        return _safe_predict(predict), mse
    except Exception:
        return None, np.inf


def _fit_cubic_spline(X, y, s=0.0):
    """
    Cubic spline interpolation. s=0 gives exact interpolation.
    """
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        unique_x, inverse, counts = np.unique(xs, return_inverse=True, return_counts=True)
        if len(unique_x) < len(xs):
            y_avg = np.zeros(len(unique_x))
            np.add.at(y_avg, inverse, ys)
            y_avg /= counts
            xs, ys = unique_x, y_avg
        
        if len(xs) < 4:
            return None, np.inf
        
        interp = UnivariateSpline(xs, ys, s=s)
        
        pred = interp(xs)
        mse = np.mean((pred - ys) ** 2)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            preds = interp(xp)
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        
        return _safe_predict(predict), mse
    except Exception:
        return None, np.inf


def _fit_fourier_series(X, y, n_harmonics=50):
    """
    Fourier series fit - good for periodic functions.
    """
    try:
        x = X[:, 0]
        n = len(x)
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min
        if x_range < 1e-12:
            return None, np.inf
        
        # Try different fundamental frequencies
        best_mse = np.inf
        best_predict = None
        
        for freq_scale in [0.5, 1.0, 2.0, 0.25, 4.0, np.pi, 2*np.pi]:
            if _remaining() < 1.0:
                break
            
            fundamental = freq_scale * 2.0 * np.pi / x_range
            cols = [np.ones(n)]
            for k in range(1, n_harmonics + 1):
                cols.append(np.sin(k * fundamental * (x - x_min)))
                cols.append(np.cos(k * fundamental * (x - x_min)))
            
            F = np.column_stack(cols)
            if not np.all(np.isfinite(F)):
                continue
            
            p = F.shape[1]
            ridge = 1e-8
            A = F.T @ F + ridge * np.eye(p)
            b = F.T @ y
            
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            pred = F @ w
            if not np.all(np.isfinite(pred)):
                continue
            
            mse = np.mean((pred - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                fund = fundamental
                nh = n_harmonics
                w_cached = w
                best_predict = _safe_predict(lambda Xp, f=fund, m=nh, wc=w_cached, xm=x_min, xr=x_range:
                    np.column_stack([np.ones(Xp.shape[0])] + 
                    [np.sin(k*f*(Xp[:,0]-xm)) for k in range(1, m+1)] + 
                    [np.cos(k*f*(Xp[:,0]-xm)) for k in range(1, m+1)]) @ wc)
        
        if best_predict is not None:
            return best_predict, best_mse
        return None, np.inf
    except Exception:
        return None, np.inf


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    n, d = X_train.shape
    
    best_predict = None
    best_mse = np.inf
    
    # Constant baseline
    const = np.mean(y_train)
    const_mse = np.mean((y_train - const) ** 2)
    best_mse = const_mse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))
    
    if d == 1 and _remaining() > 5.0:
        # 1. PCHIP interpolation - fast, exact for noiseless 1D
        pchip_pred, pchip_mse = _fit_pchip_interp(X_train, y_train)
        if pchip_pred is not None and pchip_mse < best_mse:
            best_mse = pchip_mse
            best_predict = pchip_pred
        
        # 2. Cubic spline (exact interpolation)
        if _remaining() > 5.0:
            spline_pred, spline_mse = _fit_cubic_spline(X_train, y_train, s=0.0)
            if spline_pred is not None and spline_mse < best_mse:
                best_mse = spline_mse
                best_predict = spline_pred
        
        # 3. Chebyshev collocation (exact for smooth functions)
        if _remaining() > 5.0:
            # Try different degrees
            for deg in [n - 1, min(n - 1, 80), 60, 40]:
                if _remaining() < 2.0:
                    break
                ch_pred, ch_mse = _fit_chebyshev_collocation(X_train, y_train, max_degree=deg)
                if ch_pred is not None and ch_mse < best_mse:
                    best_mse = ch_mse
                    best_predict = ch_pred
        
        # 4. Rich basis linear regression
        if _remaining() > 5.0:
            rb_pred, rb_mse = _fit_rich_basis(X_train, y_train)
            if rb_pred is not None and rb_mse < best_mse:
                best_mse = rb_mse
                best_predict = rb_pred
        
        # 5. Fourier series
        if _remaining() > 5.0:
            for nh in [30, 50, 80]:
                if _remaining() < 2.0:
                    break
                fs_pred, fs_mse = _fit_fourier_series(X_train, y_train, n_harmonics=nh)
                if fs_pred is not None and fs_mse < best_mse:
                    best_mse = fs_mse
                    best_predict = fs_pred
    
    # Final safety
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))
    
    return best_predict
# EVOLVE-BLOCK-END
