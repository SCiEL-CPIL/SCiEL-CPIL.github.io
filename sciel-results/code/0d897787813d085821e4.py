import time
import numpy as np
from scipy.interpolate import CubicSpline, make_lsq_spline
from scipy.optimize import curve_fit

def fit(X_train, y_train):
    """
    Fit a regression model.
    Strategy:
    1. Try Cubic Spline Interpolation (best for smooth noiseless data).
    2. Try a rich library of symbolic forms using curve_fit with multiple restarts.
    3. Fallback to RBF or Mean.
    """
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0].copy()
    
    # Handle potential NaNs or infinities
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

    def remaining():
        return deadline - time.time()
    
    # Sort by x for interpolation
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Remove duplicates for interpolation
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]

    best_model = None
    best_rss = np.inf
    best_score = np.inf # Normalized MSE proxy
    best_model_params = None

    var_y = np.var(y)
    if var_y < 1e-12:
        var_y = 1e-10

    # 1. Try Cubic Spline Interpolation
    # This is extremely strong for smooth, noiseless 1D data.
    try:
        if len(x_unique) > 1:
            # Use a not-a-knot boundary condition for better global smoothness
            # but for interpolation, default 'not-a-knot' is usually fine.
            # However, 'natural' might be safer if the function is bounded?
            # Actually, for interpolation of noiseless data, the spline passes through points.
            # The main risk is oscillation between points if the function is complex.
            # But with 200 points, it should be fine for most smooth functions.
            
            # Let's try a few spline types and pick the best one based on a holdout or just train error?
            # Since data is noiseless, train error is 0 for interpolation.
            # We need to check for numerical stability or oscillation.
            # A simple check: evaluate at midpoints and see if values are reasonable (e.g., within range of y).
            
            # Let's stick to the standard CubicSpline first.
            spline = CubicSpline(x_unique, y_unique)
            pred_spline = spline(x_unique)
            rss_spline = np.sum((y_unique - pred_spline)**2)
            
            norm_mse_spline = rss_spline / (len(y_unique) * var_y)
            
            if norm_mse_spline < best_score:
                best_score = norm_mse_spline
                best_model = 'spline'
                best_model_params = {'spline': spline}
                
            # If spline fits perfectly (or near perfectly), we can stop early
            if best_score < 1e-8:
                spl = best_model_params['spline']
                def predict(X_in):
                    X_in = np.asarray(X_in, dtype=np.float64)
                    xi = X_in[:, 0]
                    try:
                        preds = spl(xi)
                        preds = np.asarray(preds, dtype=np.float64)
                        if not np.all(np.isfinite(preds)):
                            preds = np.where(np.isfinite(preds), preds, np.median(y))
                        return preds
                    except Exception:
                        return np.full(len(X_in), np.median(y))
                return predict
    except Exception:
        pass

    # 2. Try Symbolic Regression
    sym_lib = _build_symbolic_library()
    
    for name, func, n_params in sym_lib:
        if remaining() < 3:
            break
            
        try:
            p0_list = _initial_params_for_form(func, n_params, x, y)
            
            best_local_rss = np.inf
            best_local_params = None
            
            for p0 in p0_list:
                if remaining() < 1:
                    break
                try:
                    popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
                    pred = func(x, *popt)
                    pred = np.asarray(pred, dtype=np.float64)
                    if not np.all(np.isfinite(pred)):
                        continue
                    rss = np.sum((y - pred)**2)
                    if rss < best_local_rss:
                        best_local_rss = rss
                        best_local_params = popt
                except Exception:
                    continue
            
            if best_local_rss < np.inf:
                norm_mse_nl = best_local_rss / (n * var_y)
                if norm_mse_nl < best_score:
                    best_score = norm_mse_nl
                    best_model = 'nonlinear'
                    best_model_params = {'func': func, 'params': best_local_params}
                    
        except Exception:
            continue

    # 3. Try RBF Network (Fallback if symbolic and spline fail)
    try:
        n_centers = len(x_unique)
        if n_centers > 100:
            indices = np.linspace(0, n_centers - 1, 100).astype(int)
            centers = x_unique[indices]
            y_centers = y_unique[indices]
        else:
            centers = x_unique
            y_centers = y_unique
            
        if len(centers) > 1:
            diffs = np.diff(centers)
            eps = np.mean(diffs)
        else:
            eps = 1.0
            
        best_rbf_score = np.inf
        best_rbf_weights = None
        best_rbf_eps = None
        
        for eps_factor in [0.5, 1.0, 2.0, 5.0]:
            eps_try = eps * eps_factor
            if eps_try < 1e-6:
                continue
                
            K = np.exp(-eps_try * (x_unique[:, np.newaxis] - centers[np.newaxis, :])**2)
            
            try:
                weights, *_ = np.linalg.lstsq(K, y_unique, rcond=None)
                pred_rbf = K @ weights
                rss_rbf = np.sum((y_unique - pred_rbf)**2)
                norm_mse_rbf = rss_rbf / (len(y_unique) * var_y)
                
                if norm_mse_rbf < best_rbf_score:
                    best_rbf_score = norm_mse_rbf
                    best_rbf_weights = weights
                    best_rbf_eps = eps_try
                    
            except Exception:
                continue
        
        if best_rbf_score < best_score:
            best_score = best_rbf_score
            best_model = 'rbf'
            best_model_params = {
                'centers': centers,
                'weights': best_rbf_weights,
                'eps': best_rbf_eps
            }
            
    except Exception:
        pass

    # 4. Fallback: Mean
    if best_model is None:
        best_model = 'mean'
        best_model_params = {'mean': np.mean(y)}

    # 5. Construct Predictor
    
    if best_model == 'spline':
        spline = best_model_params['spline']
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                preds = spline(xi)
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, np.median(y))
                return preds
            except Exception:
                return np.full(len(X_in), np.median(y))

    elif best_model == 'rbf':
        centers = best_model_params['centers']
        weights = best_model_params['weights']
        eps = best_model_params['eps']
        
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                K_pred = np.exp(-eps * (xi[:, np.newaxis] - centers[np.newaxis, :])**2)
                preds = K_pred @ weights
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, np.median(y))
                return preds
            except Exception:
                return np.full(len(X_in), np.median(y))

    elif best_model == 'nonlinear':
        func = best_model_params['func']
        params = best_model_params['params']
        
        def predict(X_in):
            X_in = np.asarray(X_in, dtype=np.float64)
            xi = X_in[:, 0]
            try:
                preds = func(xi, *params)
                preds = np.asarray(preds, dtype=np.float64)
                if not np.all(np.isfinite(preds)):
                    preds = np.where(np.isfinite(preds), preds, np.median(y))
                return preds
            except Exception:
                return np.full(len(X_in), np.median(y))
                
    else:
        mean_y = best_model_params['mean']
        def predict(X_in):
            return np.full(len(X_in), mean_y)

    return predict

def _build_symbolic_library():
    """
    Build a structured library of candidate 1D symbolic forms.
    Each entry: (name, func, n_params)
    """
    lib = []
    
    # Polynomials
    lib.append(("poly1", lambda x, a, b: a*x + b, 2))
    lib.append(("poly2", lambda x, a, b, c: a*x**2 + b*x + c, 3))
    lib.append(("poly3", lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, 4))
    lib.append(("poly4", lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e, 5))
    
    # Trig
    lib.append(("sin1", lambda x, a, b, c: a*np.sin(b*x) + c, 3))
    lib.append(("cos1", lambda x, a, b, c: a*np.cos(b*x) + c, 3))
    lib.append(("sin2", lambda x, a, b, c, d: a*np.sin(b*x + c) + d, 4))
    lib.append(("cos2", lambda x, a, b, c, d: a*np.cos(b*x + c) + d, 4))
    lib.append(("sincos", lambda x, a, b, c, d, e: a*np.sin(b*x) + c*np.cos(d*x) + e, 5))
    
    # Exp
    lib.append(("exp1", lambda x, a, b, c: a*np.exp(b*x) + c, 3))
    lib.append(("expneg", lambda x, a, b, c: a*np.exp(-b*x) + c, 3))
    lib.append(("gauss", lambda x, a, b, c, d: a*np.exp(-b*(x-c)**2) + d, 4))
    
    # Log
    lib.append(("log1", lambda x, a, b, c: a*np.log(b + np.abs(x)) + c, 3))
    lib.append(("log2", lambda x, a, b, c: a*np.log(b + x) + c, 3))
    
    # Power
    lib.append(("pow1", lambda x, a, b, c: a * np.sign(x) * np.abs(x)**b + c, 3))
    lib.append(("pow2", lambda x, a, b, c, d: a * np.sign(x) * np.abs(x)**b + c*x + d, 4))
    
    # Rational
    lib.append(("rat1", lambda x, a, b, c: a*x/(b + x) + c, 3))
    lib.append(("rat2", lambda x, a, b, c, d: a*x/(b + c*x) + d, 4))
    lib.append(("rat3", lambda x, a, b, c: a / (b + x) + c, 3))
    lib.append(("rat4", lambda x, a, b, c, d: a / (b + c*x) + d, 4))
    
    # Hyperbolic
    lib.append(("tanh1", lambda x, a, b, c: a*np.tanh(b*x) + c, 3))
    lib.append(("tanh2", lambda x, a, b, c, d: a*np.tanh(b*x + c) + d, 4))
    
    # Sigmoid
    lib.append(("sig1", lambda x, a, b, c: a / (1 + np.exp(-b*x)) + c, 3))
    
    # Absolute value based
    lib.append(("abs1", lambda x, a, b, c: a*np.abs(x) + b*x + c, 3))
    lib.append(("abs2", lambda x, a, b, c, d: a*np.abs(x - b) + c*x + d, 4))
    
    # Composites
    lib.append(("sinpoly", lambda x, a, b, c, d, e: a*np.sin(b*x) + c*x**2 + d*x + e, 5))
    lib.append(("exppoly", lambda x, a, b, c, d: a*np.exp(-b*x**2) + c*x + d, 4))
    
    return lib

def _initial_params_for_form(func, n_params, x, y):
    """Generate multiple initial parameter guesses for a given form."""
    y_mean = np.mean(y)
    y_std = np.std(y) if np.std(y) > 1e-12 else 1.0
    x_range = np.max(x) - np.min(x)
    if x_range < 1e-12:
        x_range = 1.0
    x_mean = np.mean(x)
    
    guesses = []
    
    # Generic heuristics based on scale
    guesses.append(np.full(n_params, 1.0))
    guesses.append([y_std] + [1.0]*(n_params-2) + [y_mean])
    
    if n_params == 2:
        guesses.append([y_std / x_range, y_mean])
    elif n_params == 3:
        guesses.append([y_std, 1.0, y_mean])
        guesses.append([y_std, 2*np.pi/x_range, y_mean])
        guesses.append([y_std, 1.0/x_range, y_mean])
        guesses.append([y_std, 0.5, y_mean])
        guesses.append([y_std, -1.0, y_mean])
    elif n_params == 4:
        guesses.append([y_std, 1.0, 0.0, y_mean])
        guesses.append([y_std, 2*np.pi/x_range, 0.0, y_mean])
        guesses.append([y_std, 1.0, x_mean, y_mean])
        guesses.append([y_std, 1.0/x_range, 0.0, y_mean])
    elif n_params == 5:
        guesses.append([y_std, 1.0, 0.0, y_std, y_mean])
        guesses.append([y_std, 2*np.pi/x_range, 0.0, y_std/x_range, y_mean])
        guesses.append([y_std, 1.0, 0.0, 0.0, y_mean])
        guesses.append([y_std, 1.0, 0.0, y_std, y_mean])
    
    return guesses
