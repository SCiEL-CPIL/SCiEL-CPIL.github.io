"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
from scipy.optimize import differential_evolution, minimize
from scipy.linalg import lstsq

def fit(X_train, y_train):
    """
    Symbolic regression for 1D noiseless data.
    Strategy:
    1. Fast structural search for simple functional forms (a*f(b*x+c)+d).
    2. If found with high precision, return it.
    3. Otherwise, fall back to Cubic Spline interpolation (exact on training points, smooth).
    """
    start_time = time.time()
    DEADLINE = 300
    SAFETY_MARGIN = 10
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if X.ndim == 2 and X.shape[1] == 1:
        x = X.flatten()
    elif X.ndim == 1:
        x = X
    else:
        # Fallback for multi-dim (should not happen per contract)
        # Use linear regression on all features
        try:
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            def predict_md(X_test):
                return X_test @ w
            return predict_md
        except:
            def predict_mean(X_test):
                return np.full(len(X_test), np.mean(y))
            return predict_mean
    
    # Sort by x for interpolation fallback
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicates (though noiseless uniform likely means unique)
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    if len(unique_x) < len(x_sorted):
        y_unique = np.zeros_like(unique_x)
        for i, idx in enumerate(unique_indices):
            mask = (x_sorted == unique_x[i])
            y_unique[i] = np.mean(y_sorted[mask])
        x_sorted = unique_x
        y_sorted = y_unique

    y_var = np.var(y)
    if y_var < 1e-12:
        y_var = 1.0 # Avoid division by zero
    
    # Candidate functions for structural search
    # We look for y = a * f(b * x + c) + d
    candidate_funcs = [
        ("identity", lambda t: t),
        ("square", lambda t: t ** 2),
        ("cube", lambda t: t ** 3),
        ("sin", lambda t: np.sin(t)),
        ("cos", lambda t: np.cos(t)),
        ("exp", lambda t: np.exp(t)),
        ("exp_neg", lambda t: np.exp(-t)),
        ("tanh", lambda t: np.tanh(t)),
        ("sigmoid", lambda t: 1.0 / (1.0 + np.exp(-t))),
        ("log", lambda t: np.log(np.abs(t) + 1e-10)),
        ("sqrt", lambda t: np.sqrt(np.abs(t))),
        ("inv", lambda t: 1.0 / (t + 1e-10)),
        ("gauss", lambda t: np.exp(-t ** 2)),
        ("ramp", lambda t: np.maximum(t, 0)),
    ]
    
    best_mse = np.inf
    best_model = None
    best_name = "none"
    
    # Time budget for search: 60 seconds max
    search_deadline = start_time + 60
    
    for name, f in candidate_funcs:
        if time.time() > search_deadline:
            break
            
        # Objective: minimize MSE for params (a, b, c, d)
        def obj(params):
            a, b, c, d = params
            try:
                val = a * f(b * x + c) + d
                if not np.all(np.isfinite(val)):
                    return 1e100
                return np.mean((val - y) ** 2)
            except:
                return 1e100
        
        # Initial guesses
        # a ~ y_range, b ~ 1/range, c ~ 0, d ~ y_mean
        y_range = np.ptp(y)
        x_range = np.ptp(x)
        if x_range < 1e-10: x_range = 1.0
        
        # Try a few different starting points for local optimization (Nelder-Mead)
        # This is faster than DE for smooth 4D landscapes
        starts = [
            [y_range, 1.0/x_range, 0.0, np.mean(y)],
            [y_range, 2.0/x_range, 0.0, np.mean(y)],
            [y_range, 0.5/x_range, 0.0, np.mean(y)],
            [-y_range, 1.0/x_range, 0.0, np.mean(y)],
            [y_range, 1.0/x_range, -x.mean(), np.mean(y)],
            [y_range, 1.0/x_range, x.mean(), np.mean(y)],
        ]
        
        for s in starts:
            if time.time() > search_deadline:
                break
            try:
                res = minimize(obj, s, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_mse:
                    best_mse = res.fun
                    best_model = (name, f, res.x)
                    best_name = name
                    
                    # Check for near-perfect fit
                    if best_mse / y_var < 1e-10:
                        break
            except:
                continue
        
        if best_mse / y_var < 1e-10:
            break

    # Check if we found a good symbolic fit
    if best_model is not None and best_mse / y_var < 1e-8:
        name, f, params = best_model
        a, b, c, d = params
        
        def predict_symbolic(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 2 and X_test.shape[1] == 1:
                x_test = X_test.flatten()
            else:
                x_test = X_test.flatten()
            val = a * f(b * x_test + c) + d
            return np.nan_to_num(val, nan=0.0, posinf=1e10, neginf=-1e10)
            
        return predict_symbolic

    # Fallback: Cubic Spline Interpolation
    # This is exact on training points and smooth.
    # For noiseless data, this is a very strong baseline.
    # We use 'not-a-knot' boundary conditions which are often better than 'natural' 
    # for functions that don't necessarily have zero curvature at boundaries.
    
    try:
        # CubicSpline requires strictly increasing x
        # We already sorted and uniqued.
        spline = CubicSpline(x_sorted, y_sorted, bc_type='not-a-knot')
        
        def predict_spline(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 2 and X_test.shape[1] == 1:
                x_test = X_test.flatten()
            else:
                x_test = X_test.flatten()
            pred = spline(x_test)
            return np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
            
        # Verify finite on training data
        test_pred = predict_spline(X_train[:10])
        if np.all(np.isfinite(test_pred)):
            return predict_spline
        else:
            raise ValueError("Non-finite spline output")
            
    except Exception:
        # Fallback to PCHIP (monotone cubic) if spline fails (e.g. oscillations)
        try:
            pchip = PchipInterpolator(x_sorted, y_sorted)
            
            def predict_pchip(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 2 and X_test.shape[1] == 1:
                    x_test = X_test.flatten()
                else:
                    x_test = X_test.flatten()
                pred = pchip(x_test)
                return np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
                
            return predict_pchip
        except Exception:
            # Fallback to Linear Interpolation
            try:
                interp = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
                
                def predict_linear(X_test):
                    X_test = np.asarray(X_test, dtype=np.float64)
                    if X_test.ndim == 2 and X_test.shape[1] == 1:
                        x_test = X_test.flatten()
                    else:
                        x_test = X_test.flatten()
                    pred = interp(x_test)
                    return np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
                    
                return predict_linear
            except Exception:
                # Last resort: Mean
                mean_y = np.mean(y)
                def predict_mean(X_test):
                    return np.full(len(X_test), mean_y)
                return predict_mean

# EVOLVE-BLOCK-END
