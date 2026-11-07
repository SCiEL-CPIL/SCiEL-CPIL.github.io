import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Flatten for 1D processing
    x = X.flatten()
    y_flat = y.flatten()
    
    # Check for constant y
    if np.var(y_flat) < 1e-10:
        mean_y = np.mean(y_flat)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    # Handle duplicates
    unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) != len(x_sorted):
        y_sum = np.zeros_like(unique_x, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        x_sorted = unique_x
    
    if len(x_sorted) < 2:
        mean_y = np.mean(y_flat)
        def predict_single(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_single
    
    x_min, x_max = x_sorted[0], x_sorted[-1]
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        mean_y = np.mean(y_flat)
        def predict_single(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_single
    
    # Normalize to [-1, 1] for stability
    x_norm = (x_sorted - (x_max + x_min) / 2) / (x_range / 2)
    
    # Strategy: Try multiple basis functions and select the best via validation
    # We'll use a hold-out set for selection
    
    n = len(x_sorted)
    # Use the last 10% for validation if possible, otherwise just use all for fitting
    # Since data is noiseless, interpolation should be perfect if the basis is correct.
    # However, high-degree polynomials can be unstable.
    # PCHIP is robust for monotonic-ish data.
    # Let's try a few candidates:
    # 1. PCHIP
    # 2. Cubic Spline
    # 3. Chebyshev (low degree)
    
    candidates = []
    
    # Candidate 1: PCHIP
    try:
        if np.any(np.diff(x_sorted) <= 0):
            raise ValueError("Non-increasing X")
        pchip_interp = PchipInterpolator(x_sorted, y_sorted)
        # Test for finiteness
        test_x = x_sorted[::max(1, n//10)]
        test_y = pchip_interp(test_x)
        if np.all(np.isfinite(test_y)):
            candidates.append(('pchip', pchip_interp))
    except Exception:
        pass
    
    # Candidate 2: Cubic Spline
    try:
        spline = CubicSpline(x_sorted, y_sorted)
        test_x = x_sorted[::max(1, n//10)]
        test_y = spline(test_x)
        if np.all(np.isfinite(test_y)):
            candidates.append(('spline', spline))
    except Exception:
        pass
    
    # Candidate 3: Chebyshev fit
    # We need to choose a degree. Since it's noiseless, a good fit should have near-zero error on training data.
    # But we can't just pick max degree due to Runge phenomenon.
    # Let's try degrees 10, 20, 30 and pick the one with lowest training error (which should be ~0 for all if they fit well,
    # but lower degrees are more stable).
    # Actually, since it's noiseless, if the function is smooth, a moderate degree Chebyshev should be excellent.
    # Let's pick the smallest degree that achieves a very low training error.
    
    best_cheb_c = None
    best_cheb_deg = -1
    min_train_err = float('inf')
    
    # Try a range of degrees
    for deg in [5, 10, 15, 20, 30, 50]:
        if time.time() > deadline:
            break
        try:
            c = np.polynomial.chebyshev.chebfit(x_norm, y_sorted, deg)
            y_pred = np.polynomial.chebyshev.chebval(x_norm, c)
            err = np.mean((y_pred - y_sorted)**2)
            if err < min_train_err:
                min_train_err = err
                best_cheb_c = c
                best_cheb_deg = deg
            # If error is very small, we can stop early as higher degrees might just add instability
            if err < 1e-10:
                break
        except Exception:
            continue
    
    if best_cheb_c is not None:
        # Wrap Chebyshev in a callable interface compatible with the others
        class ChebWrapper:
            def __init__(self, c, x_min, x_max, x_range):
                self.c = c
                self.x_min = x_min
                self.x_max = x_max
                self.x_range = x_range
            
            def __call__(self, X_eval):
                X_eval = np.asarray(X_eval)
                X_norm = (X_eval - (self.x_max + self.x_min) / 2) / (self.x_range / 2)
                return np.polynomial.chebyshev.chebval(X_norm, self.c)
        
        candidates.append(('cheb', ChebWrapper(best_cheb_c, x_min, x_max, x_range)))
    
    # Fallback: Linear Interpolation
    candidates.append(('linear', None)) # Special case
    
    # Selection Strategy:
    # Since data is noiseless, we expect the correct model to have ~0 error on training data.
    # However, PCHIP and Spline are interpolants, so they have 0 error by construction (at the knots).
    # Chebyshev is an approximation.
    # The risk is oscillation between knots for high-degree polynomials or splines.
    # PCHIP is designed to avoid overshoot.
    # Let's use a simple heuristic:
    # If PCHIP is available, it's generally a safe bet for 1D noiseless data unless the function is very oscillatory.
    # If the function is highly oscillatory, Chebyshev might be better if the degree is high enough.
    
    # Let's score them on a validation set (last 10% of points) to see which generalizes best.
    # But wait, if the model is an interpolant, validation error is 0 if the validation points are in the training set?
    # No, the validation points are a subset of the training points if we just split indices.
    # To properly validate, we should hold out some points.
    
    # Let's hold out the last 20 points for validation.
    n_val = min(20, n // 10)
    if n_val > 0:
        x_val = x_sorted[-n_val:]
        y_val = y_sorted[-n_val:]
        x_fit = x_sorted[:-n_val]
        y_fit = y_sorted[:-n_val]
        
        # Refit candidates on x_fit, y_fit
        scored_candidates = []
        
        # 1. PCHIP
        try:
            if np.any(np.diff(x_fit) <= 0):
                raise ValueError("Non-increasing X")
            pchip_val = PchipInterpolator(x_fit, y_fit)
            y_pred_val = pchip_val(x_val)
            if np.all(np.isfinite(y_pred_val)):
                err = np.mean((y_pred_val - y_val)**2)
                scored_candidates.append((err, 'pchip', pchip_val))
        except Exception:
            pass
        
        # 2. Cubic Spline
        try:
            spline_val = CubicSpline(x_fit, y_fit)
            y_pred_val = spline_val(x_val)
            if np.all(np.isfinite(y_pred_val)):
                err = np.mean((y_pred_val - y_val)**2)
                scored_candidates.append((err, 'spline', spline_val))
        except Exception:
            pass
        
        # 3. Chebyshev
        best_cheb_c_fit = None
        best_cheb_deg_fit = -1
        min_fit_err = float('inf')
        
        # Normalize x_fit
        x_fit_min, x_fit_max = x_fit[0], x_fit[-1]
        x_fit_range = x_fit_max - x_fit_min
        if x_fit_range > 0:
            x_fit_norm = (x_fit - (x_fit_max + x_fit_min) / 2) / (x_fit_range / 2)
            x_val_norm = (x_val - (x_fit_max + x_fit_min) / 2) / (x_fit_range / 2)
            
            for deg in [5, 10, 15, 20, 30, 50]:
                if time.time() > deadline:
                    break
                try:
                    c = np.polynomial.chebyshev.chebfit(x_fit_norm, y_fit, deg)
                    y_pred_fit = np.polynomial.chebyshev.chebval(x_fit_norm, c)
                    fit_err = np.mean((y_pred_fit - y_fit)**2)
                    if fit_err < min_fit_err:
                        min_fit_err = fit_err
                        best_cheb_c_fit = c
                        best_cheb_deg_fit = deg
                    if fit_err < 1e-10:
                        break
                except Exception:
                    continue
            
            if best_cheb_c_fit is not None:
                y_pred_val = np.polynomial.chebyshev.chebval(x_val_norm, best_cheb_c_fit)
                if np.all(np.isfinite(y_pred_val)):
                    err = np.mean((y_pred_val - y_val)**2)
                    # Wrap for consistency
                    class ChebWrapperVal:
                        def __init__(self, c, x_min, x_max, x_range):
                            self.c = c
                            self.x_min = x_min
                            self.x_max = x_max
                            self.x_range = x_range
                        
                        def __call__(self, X_eval):
                            X_eval = np.asarray(X_eval)
                            X_norm = (X_eval - (self.x_max + self.x_min) / 2) / (self.x_range / 2)
                            return np.polynomial.chebyshev.chebval(X_norm, self.c)
                    
                    scored_candidates.append((err, 'cheb', ChebWrapperVal(best_cheb_c_fit, x_fit_min, x_fit_max, x_fit_range)))
        
        # 4. Linear Interpolation
        y_pred_val = np.interp(x_val, x_fit, y_fit)
        err = np.mean((y_pred_val - y_val)**2)
        scored_candidates.append((err, 'linear', None))
        
        # Select best
        if scored_candidates:
            scored_candidates.sort(key=lambda item: item[0])
            best_err, best_name, best_model = scored_candidates[0]
        else:
            best_name = 'linear'
            best_model = None
    else:
        # Not enough points for validation, just pick PCHIP or Spline
        best_name = 'pchip'
        best_model = None # Will be refit on all data below

    # Refit the best model on ALL data
    if best_name == 'pchip':
        try:
            if np.any(np.diff(x_sorted) <= 0):
                raise ValueError("Non-increasing X")
            final_model = PchipInterpolator(x_sorted, y_sorted)
        except Exception:
            final_model = None
            best_name = 'linear'
    elif best_name == 'spline':
        try:
            final_model = CubicSpline(x_sorted, y_sorted)
        except Exception:
            final_model = None
            best_name = 'linear'
    elif best_name == 'cheb':
        # We need to refit Chebyshev on all data
        # Re-use the logic from above but on full data
        x_min, x_max = x_sorted[0], x_sorted[-1]
        x_range = x_max - x_min
        x_norm = (x_sorted - (x_max + x_min) / 2) / (x_range / 2)
        
        best_cheb_c_final = None
        min_train_err_final = float('inf')
        
        for deg in [5, 10, 15, 20, 30, 50]:
            if time.time() > deadline:
                break
            try:
                c = np.polynomial.chebyshev.chebfit(x_norm, y_sorted, deg)
                y_pred = np.polynomial.chebyshev.chebval(x_norm, c)
                err = np.mean((y_pred - y_sorted)**2)
                if err < min_train_err_final:
                    min_train_err_final = err
                    best_cheb_c_final = c
                if err < 1e-10:
                    break
            except Exception:
                continue
        
        if best_cheb_c_final is not None:
            class ChebWrapperFinal:
                def __init__(self, c, x_min, x_max, x_range):
                    self.c = c
                    self.x_min = x_min
                    self.x_max = x_max
                    self.x_range = x_range
                
                def __call__(self, X_eval):
                    X_eval = np.asarray(X_eval)
                    X_norm = (X_eval - (self.x_max + self.x_min) / 2) / (self.x_range / 2)
                    return np.polynomial.chebyshev.chebval(X_norm, self.c)
            
            final_model = ChebWrapperFinal(best_cheb_c_final, x_min, x_max, x_range)
        else:
            final_model = None
            best_name = 'linear'
    else: # linear
        final_model = None

    def predict(X):
        X_eval = np.asarray(X).flatten()
        
        if best_name == 'linear' or final_model is None:
            y_pred = np.interp(X_eval, x_sorted, y_sorted)
        else:
            y_pred = final_model(X_eval)
        
        # Ensure finite output
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        return y_pred.reshape(np.asarray(X).shape)

    return predict
