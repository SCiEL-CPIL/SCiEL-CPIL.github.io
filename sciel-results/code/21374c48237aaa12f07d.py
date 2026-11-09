import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
import time

def fit(X_train, y_train):
    start_time = time.time()
    X = np.asarray(X_train, dtype=np.float64).ravel()
    y = np.asarray(y_train, dtype=np.float64).ravel()
    
    # Sort by X
    order = np.argsort(X)
    Xs = X[order]
    ys = y[order]
    
    # Remove duplicate X values (keep mean y)
    unique_X, inverse, counts = np.unique(Xs, return_inverse=True, return_counts=True)
    if len(unique_X) < len(Xs):
        ys_sum = np.zeros(len(unique_X))
        np.add.at(ys_sum, inverse, ys)
        ys = ys_sum / counts
        Xs = unique_X
    else:
        Xs = Xs
        ys = ys
    
    n = len(Xs)
    
    # Try to fit a spline interpolant
    spline = None
    pchip = None
    
    if n >= 4:
        try:
            spline = CubicSpline(Xs, ys, bc_type='not-a-knot')
        except Exception:
            spline = None
    
    if n >= 2:
        try:
            pchip = PchipInterpolator(Xs, ys)
        except Exception:
            pchip = None
    
    # Determine range
    x_min, x_max = Xs[0], Xs[-1]
    
    # Also fit a fallback polynomial (degree 3) for extrapolation
    try:
        coeffs = np.polyfit(Xs, ys, 3)
    except Exception:
        coeffs = np.array([0.0, 0.0, 0.0, np.mean(ys)])
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        orig_shape = X_test.shape
        X_flat = X_test.ravel()
        y_pred = np.empty(len(X_flat), dtype=np.float64)
        
        for i, x in enumerate(X_flat):
            if not np.isfinite(x):
                y_pred[i] = 0.0
                continue
            
            if x < x_min or x > x_max:
                # Extrapolation: use polynomial fallback
                val = np.polyval(coeffs, x)
                if not np.isfinite(val):
                    if x < x_min:
                        val = ys[0]
                    else:
                        val = ys[-1]
                y_pred[i] = val
            else:
                # Interpolation
                val = None
                if spline is not None:
                    try:
                        val = float(spline(x))
                    except Exception:
                        val = None
                if val is None or not np.isfinite(val):
                    if pchip is not None:
                        try:
                            val = float(pchip(x))
                        except Exception:
                            val = None
                if val is None or not np.isfinite(val):
                    # Last resort: linear interpolation
                    idx = np.searchsorted(Xs, x)
                    if idx == 0:
                        val = float(ys[0])
                    elif idx >= n:
                        val = float(ys[-1])
                    else:
                        x0, x1 = Xs[idx-1], Xs[idx]
                        y0, y1 = ys[idx-1], ys[idx]
                        if x1 > x0:
                            t = (x - x0) / (x1 - x0)
                            val = y0 + t * (y1 - y0)
                        else:
                            val = y0
                
                y_pred[i] = val
        
        return y_pred.reshape(orig_shape)
    
    return predict
