import time
import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import norm
import itertools

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 295  # 5s safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0].copy()
    
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    var_y = np.var(y)
    if var_y < 1e-12:
        var_y = 1e-10
    
    median_y = np.median(y)
    y_mean = np.mean(y)
    y_std = np.std(y) if np.std(y) > 1e-12 else 1.0
    
    def remaining():
        return deadline - time.time()
    
    # Sort x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Get unique x values
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]
    n_unique = len(x_unique)
    
    best_rss = np.inf
    best_predictor = None
    best_name = "none"
    
    # Strategy 1: Cubic Spline (very fast, good for smooth functions)
    try:
        if n_unique >= 4:
            spline = CubicSpline(x_unique, y_unique)
            pred_train = spline(x_unique)
            rss_spline = np.sum((y_unique - pred_train)**2)
            if rss_spline < best_rss:
                best_rss = rss_spline
                best_name = "spline"
                
                def predict_spline(X_in):
                    X_in = np.asarray(X_in, dtype=np.float64)
                    xi = X_in[:, 0]
                    try:
                        preds = spline(xi)
                        preds = np.asarray(preds, dtype=np.float64)
                        if not np.all(np.isfinite(preds)):
                            preds = np.where(np.isfinite(preds), preds, median_y)
                        return preds
                    except Exception:
                        return np.full(len(X_in), median_y)
                
                best_predictor = predict_spline
    except Exception:
        pass
    
    # Strategy 2: Gaussian Process / Kernel regression with multiple length scales
    if remaining() > 2:
        try:
            # Use a subset of points for efficiency if too many
            if n_unique > 100:
                idx_subset = np.linspace(0, n_unique - 1, 100).astype(int)
                x_sub = x_unique[idx_subset]
                y_sub = y_unique[idx_subset]
            else:
                x_sub = x_unique
                y_sub = y_unique
            
            # Try multiple kernel bandwidths
            x_range = x_sub.max() - x_sub.min()
            if x_range < 1e-12:
                x_range = 1.0
            
            best_kernel_rss = np.inf
            best_kernel_params = None
            
            # Length scales to try
            length_scales = [x_range * f for f in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
            # Also try based on point spacing
            if len(x_sub) > 2:
                median_spacing = np.median(np.diff(x_sub))
                length_scales.extend([median_spacing * f for f in [0.5, 1.0, 2.0, 5.0]])
            
            for ls in length_scales:
                if remaining() < 1:
                    break
                if ls < 1e-10:
                    continue
                
                # RBF kernel
                K = np.exp(-0.5 * (x_sub[:, np.newaxis] - x_sub[np.newaxis, :])**2 / (ls**2))
                # Add small regularization for stability
                K_reg = K + 1e-8 * np.eye(len(x_sub))
                
                try:
                    # Solve K @ alpha = y
                    alpha = np.linalg.solve(K_reg, y_sub)
                    
                    # Evaluate on training points
                    pred_train = K @ alpha
                    rss_kernel = np.sum((y_sub - pred_train)**2)
                    
                    if rss_kernel < best_kernel_rss:
                        best_kernel_rss = rss_kernel
                        best_kernel_params = (x_sub, y_sub, alpha, ls, K)
                except Exception:
                    continue
            
            # Compare with spline
            if best_kernel_rss < best_rss and best_kernel_params is not None:
                best_rss = best_kernel_rss
                best_name = "kernel"
                x_sub, y_sub, alpha, ls, K = best_kernel_params
                
                def predict_kernel(X_in):
                    X_in = np.asarray(X_in, dtype=np.float64)
                    xi = X_in[:, 0]
                    try:
                        K_pred = np.exp(-0.5 * (xi[:, np.newaxis] - x_sub[np.newaxis, :])**2 / (ls**2))
                        preds = K_pred @ alpha
                        preds = np.asarray(preds, dtype=np.float64)
                        if not np.all(np.isfinite(preds)):
                            preds = np.where(np.isfinite(preds), preds, median_y)
                        return preds
                    except Exception:
                        return np.full(len(X_in), median_y)
                
                best_predictor = predict_kernel
        except Exception:
            pass
    
    # Strategy 3: Symbolic regression with expanded library
    if remaining() > 10:
        sym_lib = []
        
        # Polynomials
        sym_lib.append(("poly1", lambda x, a, b: a*x + b, 2))
        sym_lib.append(("poly2", lambda x, a, b, c: a*x**2 + b*x + c, 3))
        sym_lib.append(("poly3", lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, 4))
        sym_lib.append(("poly4", lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e, 5))
        sym_lib.append(("poly5", lambda x, a,b,c,d,e,f: a*x**5+b*x**4+c*x**3+d*x**2+e*x+f, 6))
        
        # Trig
        sym_lib.append(("sin1", lambda x, a, b, c: a*np.sin(b*x) + c, 3))
        sym_lib.append(("cos1", lambda x, a, b, c: a*np.cos(b*x) + c, 3))
        sym_lib.append(("sin2", lambda x, a, b, c, d: a*np.sin(b*x + c) + d, 4))
        sym_lib.append(("cos2", lambda x, a, b, c, d: a*np.cos(b*x + c) + d, 4))
        sym_lib.append(("sincos", lambda x, a, b, c, d, e: a*np.sin(b*x) + c*np.cos(d*x) + e, 5))
        sym_lib.append(("sin2x", lambda x, a, b, c, d: a*np.sin(2*b*x) + c*x + d, 4))
        sym_lib.append(("sincos2", lambda x, a,b,c,d,e,f: a*np.sin(b*x)+c*np.cos(2*b*x)+d*x+e, 5))
        
        # Exp
        sym_lib.append(("exp1", lambda x, a, b, c: a*np.exp(b*x) + c, 3))
        sym_lib.append(("expneg", lambda x, a, b, c: a*np.exp(-b*x) + c, 3))
        sym_lib.append(("gauss", lambda x, a, b, c, d: a*np.exp(-b*(x-c)**2) + d, 4))
        sym_lib.append(("gauss2", lambda x, a, b, c, d, e: a*np.exp(-b*(x-c)**2) + d*x + e, 5))
        
        # Log
        sym_lib.append(("log1", lambda x, a, b, c: a*np.log(np.abs(x) + b) + c, 3))
        sym_lib.append(("logpos", lambda x, a, b, c: a*np.log(np.maximum(x, 0.01) + b) + c, 3))
        
        # Power
        sym_lib.append(("pow1", lambda x, a, b, c: a * np.sign(x) * np.abs(x)**b + c, 3))
        sym_lib.append(("pow2", lambda x, a, b, c, d: a * np.sign(x) * np.abs(x)**b + c*x + d, 4))
        sym_lib.append(("sqrt", lambda x, a, b, c: a*np.sqrt(np.abs(x) + b) + c, 3))
        sym_lib.append(("sqrt2", lambda x, a, b, c, d: a*np.sqrt(np.abs(x) + b) + c*x + d, 4))
        
        # Rational
        sym_lib.append(("rat1", lambda x, a, b, c: a*x/(b + x) + c, 3))
        sym_lib.append(("rat2", lambda x, a, b, c, d: a*x/(b + c*x) + d, 4))
        sym_lib.append(("rat3", lambda x, a, b, c: a / (b + x) + c, 3))
        sym_lib.append(("rat4", lambda x, a, b, c, d: a / (b + c*x) + d, 4))
        sym_lib.append(("rat5", lambda x, a, b, c, d: a*(x-b)/(c + x) + d, 4))
        
        # Tanh
        sym_lib.append(("tanh1", lambda x, a, b, c: a*np.tanh(b*x) + c, 3))
        sym_lib.append(("tanh2", lambda x, a, b, c, d: a*np.tanh(b*x + c) + d, 4))
        
        # Sigmoid
        sym_lib.append(("sig1", lambda x, a, b, c: a / (1 + np.exp(-b*x)) + c, 3))
        sym_lib.append(("sig2", lambda x, a, b, c, d: a / (1 + np.exp(-b*(x-c))) + d, 4))
        
        # Abs
        sym_lib.append(("abs1", lambda x, a, b, c: a*np.abs(x) + b*x + c, 3))
        sym_lib.append(("abs2", lambda x, a, b, c, d: a*np.abs(x - b) + c*x + d, 4))
        sym_lib.append(("abs3", lambda x, a, b, c, d, e: a*np.abs(x-b) + c*np.abs(x-d) + e, 5))
        
        # Composite
        sym_lib.append(("sinpoly", lambda x, a, b, c, d, e: a*np.sin(b*x) + c*x**2 + d*x + e, 5))
        sym_lib.append(("exppoly", lambda x, a, b, c, d: a*np.exp(-b*x**2) + c*x + d, 4))
        sym_lib.append(("exptrig", lambda x, a, b, c, d: a*np.exp(-b*x**2)*np.sin(c*x) + d, 4))
        sym_lib.append(("polytrig", lambda x, a, b, c, d, e: a*x**2 + b*np.sin(c*x) + d*x + e, 5))
        sym_lib.append(("expabs", lambda x, a, b, c, d: a*np.exp(-b*np.abs(x-c)) + d, 4))
        
        x_scale = np.max(x) - np.min(x)
        if x_scale < 1e-12:
            x_scale = 1.0
        x_mean = np.mean(x)
        
        for name, func, n_params in sym_lib:
            if remaining() < 2:
                break
            
            try:
                # Multiple initial guesses
                guesses = []
                
                # Generic random
                for _ in range(2):
                    p = np.random.uniform(-y_std, y_std, size=n_params)
                    if n_params >= 2:
                        p[-1] = y_mean
                    guesses.append(p)
                
                # Heuristic guesses
                if n_params == 2:
                    guesses.append([y_std / x_scale, y_mean])
                    guesses.append([-y_std / x_scale, y_mean])
                elif n_params == 3:
                    guesses.append([y_std, 1.0, y_mean])
                    guesses.append([y_std, 2*np.pi/x_scale, y_mean])
                    guesses.append([y_std, 1.0/x_scale, y_mean])
                    guesses.append([y_std, 0.5, y_mean])
                    guesses.append([-y_std, 1.0, y_mean])
                elif n_params == 4:
                    guesses.append([y_std, 1.0, 0.0, y_mean])
                    guesses.append([y_std, 2*np.pi/x_scale, 0.0, y_mean])
                    guesses.append([y_std, 1.0, x_mean, y_mean])
                    guesses.append([y_std, 1.0/x_scale, 0.0, y_mean])
                    guesses.append([-y_std, 1.0, 0.0, y_mean])
                elif n_params == 5:
                    guesses.append([y_std, 1.0, 0.0, y_std, y_mean])
                    guesses.append([y_std, 2*np.pi/x_scale, 0.0, y_std/x_scale, y_mean])
                    guesses.append([y_std, 1.0, 0.0, 0.0, y_mean])
                    guesses.append([-y_std, 1.0, 0.0, y_std, y_mean])
                elif n_params == 6:
                    guesses.append([y_std/x_scale**5, y_std/x_scale**4, y_std/x_scale**3, y_std/x_scale**2, y_std/x_scale, y_mean])
                
                best_local_rss = np.inf
                best_local_params = None
                
                for p0 in guesses:
                    if remaining() < 0.5:
                        break
                    try:
                        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=2000)
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
                
                if best_local_rss < np.inf and best_local_rss < best_rss:
                    best_rss = best_local_rss
                    best_name = name
                    
                    func_local = func
                    params_local = best_local_params
                    
                    def predict_sym(X_in, f=func_local, p=params_local):
                        X_in = np.asarray(X_in, dtype=np.float64)
                        xi = X_in[:, 0]
                        try:
                            preds = f(xi, *p)
                            preds = np.asarray(preds, dtype=np.float64)
                            if not np.all(np.isfinite(preds)):
                                preds = np.where(np.isfinite(preds), preds, median_y)
                            return preds
                        except Exception:
                            return np.full(len(X_in), median_y)
                    
                    best_predictor = predict_sym
                    
            except Exception:
                continue
    
    # Final safety: if no good predictor, use spline or constant
    if best_predictor is None:
        try:
            if n_unique >= 4:
                spline = CubicSpline(x_unique, y_unique)
                
                def predict_fallback(X_in):
                    X_in = np.asarray(X_in, dtype=np.float64)
                    xi = X_in[:, 0]
                    try:
                        preds = spline(xi)
                        preds = np.asarray(preds, dtype=np.float64)
                        if not np.all(np.isfinite(preds)):
                            preds = np.where(np.isfinite(preds), preds, median_y)
                        return preds
                    except Exception:
                        return np.full(len(X_in), median_y)
                
                best_predictor = predict_fallback
            else:
                def predict_fallback(X_in):
                    return np.full(len(np.asarray(X_in).shape[0]), median_y)
                best_predictor = predict_fallback
        except Exception:
            def predict_fallback(X_in):
                return np.full(len(np.asarray(X_in).shape[0]), median_y)
            best_predictor = predict_fallback
    
    return best_predictor
