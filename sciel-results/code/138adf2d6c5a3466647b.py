import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator
from numpy.polynomial import chebyshev as C

_TOTAL_DEADLINE = time.time() + 325.0

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
    """Fit PCHIP interpolator."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        # Handle duplicate x values by averaging y
        unique_xs, indices = np.unique(xs, return_index=True)
        if len(unique_xs) != len(xs):
            # Average duplicates
            ys_unique = np.zeros(len(unique_xs))
            for i, idx in enumerate(indices):
                pass # This logic is complex, better to just use np.unique with return_counts or bincount
            # Simpler: use scipy's PCHIP which handles this? No, it requires unique x.
            # Let's aggregate.
            unique_xs, inverse = np.unique(xs, return_inverse=True)
            ys_unique = np.zeros(len(unique_xs))
            counts = np.zeros(len(unique_xs))
            np.add.at(ys_unique, inverse, ys)
            np.add.at(counts, inverse, 1)
            ys_unique /= counts
            xs, ys = unique_xs, ys_unique
            
        pchip = PchipInterpolator(xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            return pchip(xp)
            
        return _safe_predict(predict)
    except Exception:
        return None

def _fit_chebyshev(X, y):
    """
    Fit Chebyshev polynomial.
    Returns (predict_fn, training_mse, degree)
    """
    try:
        x = X[:, 0]
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-12:
            return None, np.inf, 0
        
        # Map to [-1, 1]
        x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        
        # We want to find the lowest degree that fits well.
        # Try degrees from 1 to 80.
        best_degree = None
        best_w = None
        best_mse = np.inf
        
        # Since data is noiseless, we expect MSE to drop to ~0 quickly for analytic functions.
        # We iterate degrees.
        for deg in range(1, 81):
            if _remaining() < 2.0:
                break
                
            # Build Chebyshev basis
            # C.chebvander(x_norm, deg) gives T_0 ... T_deg
            F = C.chebvander(x_norm, deg)
            
            # Solve least squares: min ||F w - y||^2
            # Use lstsq for stability
            w, residuals, rank, s = np.linalg.lstsq(F, y, rcond=None)
            
            pred = F @ w
            mse = np.mean((pred - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_degree = deg
                best_w = w.copy()
            
            # Early stopping: if MSE is very small, we found the function (likely)
            # Threshold: 1e-10 relative to variance of y? Or absolute?
            y_var = np.var(y)
            if y_var > 1e-12:
                rel_mse = mse / y_var
            else:
                rel_mse = mse
            
            if rel_mse < 1e-10:
                # We have a very good fit. Stop.
                # But check if a lower degree also had this? 
                # The loop goes 1, 2, 3... so the first one to hit this is the lowest degree.
                break

        if best_w is None:
            return None, np.inf, 0

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xp_norm = 2.0 * (xp - x_min) / (x_max - x_min) - 1.0
            Fp = C.chebvander(xp_norm, best_degree)
            preds = Fp @ best_w
            return preds

        return _safe_predict(predict), best_mse, best_degree

    except Exception:
        return None, np.inf, 0

def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        
    n, d = X_train.shape
    
    # 1. Fit PCHIP
    pchip_pred = _fit_pchip(X_train, y_train)
    
    # 2. Fit Chebyshev
    ch_pred, ch_mse, ch_deg = _fit_chebyshev(X_train, y_train)
    
    # Decision Logic:
    # If Chebyshev found a low-degree fit (deg < 40) with near-zero error, prefer it.
    # Why? Chebyshev is smoother and might extrapolate better if the function is analytic.
    # If Chebyshev requires high degree (> 40) or fails, PCHIP is safer.
    
    y_var = np.var(y_train)
    if y_var < 1e-12:
        # Constant function
        const = np.mean(y_train)
        return _safe_predict(lambda Xp: np.full(Xp.shape[0], const))
        
    ch_rel_mse = ch_mse / y_var
    
    if ch_pred is not None and ch_deg < 40 and ch_rel_mse < 1e-8:
        # Good Chebyshev fit
        return ch_pred
    else:
        # Fall back to PCHIP
        if pchip_pred is not None:
            return pchip_pred
        else:
            # Last resort: Linear Interpolation
            x = X_train[:, 0]
            order = np.argsort(x)
            xs, ys = x[order], y_train[order]
            unique_xs, inverse = np.unique(xs, return_inverse=True)
            ys_unique = np.zeros(len(unique_xs))
            counts = np.zeros(len(unique_xs))
            np.add.at(ys_unique, inverse, ys)
            np.add.at(counts, inverse, 1)
            ys_unique /= counts
            
            def predict_linear(Xp):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                xp = Xp[:, 0]
                return np.interp(xp, unique_xs, ys_unique)
                
            return _safe_predict(predict_linear)
