import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import least_squares, minimize
from scipy.stats import linregress
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 310.0  # Safety margin before 330s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    X_flat = X.flatten()
    y_flat = y.flatten()
    
    n = len(X_flat)
    
    # Check if all y values are the same (constant function)
    if np.allclose(y_flat, y_flat[0]):
        c = y_flat[0]
        def predict_const(X):
            return np.full(np.asarray(X).shape, c)
        return predict_const
    
    # Sort by X for interpolation methods
    sort_idx = np.argsort(X_flat)
    X_sorted = X_flat[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    # Handle duplicate X values
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) != len(X_sorted):
        y_sum = np.zeros_like(unique_X, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        X_sorted = unique_X
        n = len(X_sorted)
    
    x_min, x_max = X_sorted.min(), X_sorted.max()
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    
    y_min, y_max = y_sorted.min(), y_sorted.max()
    y_range = y_max - y_min
    if y_range < 1e-10:
        y_range = 1.0
    
    # Normalize X to [-1, 1] for numerical stability
    x_norm = (X_sorted - (x_min + x_max) / 2) / (x_range / 2)
    
    models = []
    
    # Build rich feature matrix with diverse basis functions
    def build_features(x):
        """Build a rich set of basis functions for feature selection."""
        features = []
        names = []
        
        # Polynomials (degree 1-8)
        for d in range(1, 9):
            features.append(x ** d)
            names.append(f'poly_{d}')
        
        # Exponential
        features.append(np.exp(x))
        names.append('exp')
        features.append(np.exp(-x))
        names.append('exp_neg')
        
        # Logarithmic (safe)
        safe_x_pos = np.where(x > 0.01, x, 0.01)
        safe_x_neg = np.where(x < -0.01, x, -0.01)
        features.append(np.log(safe_x_pos))
        names.append('log_pos')
        features.append(np.log(-safe_x_neg))
        names.append('log_neg')
        
        # Trigonometric
        for freq in [0.5, 1.0, 2.0, 4.0, 8.0]:
            features.append(np.sin(freq * x))
            names.append(f'sin_{freq}')
            features.append(np.cos(freq * x))
            names.append(f'cos_{freq}')
        
        # Hyperbolic
        features.append(np.tanh(x))
        names.append('tanh')
        features.append(np.sinh(x))
        names.append('sinh')
        features.append(np.cosh(x))
        names.append('cosh')
        
        # Rational-like (safe)
        denom1 = x + 1.0
        denom1_safe = np.where(np.abs(denom1) < 0.01, 0.01 * np.sign(denom1 + 1e-10), denom1)
        features.append(1.0 / denom1_safe)
        names.append('rat_1')
        
        denom2 = x + 2.0
        denom2_safe = np.where(np.abs(denom2) < 0.01, 0.01 * np.sign(denom2 + 1e-10), denom2)
        features.append(1.0 / denom2_safe)
        names.append('rat_2')
        
        denom3 = x - 1.0
        denom3_safe = np.where(np.abs(denom3) < 0.01, 0.01 * np.sign(denom3 + 1e-10), denom3)
        features.append(1.0 / denom3_safe)
        names.append('rat_neg1')
        
        # Sqrt (safe for positive)
        safe_pos = np.where(x > 0, x, 0)
        features.append(np.sqrt(safe_pos))
        names.append('sqrt_pos')
        
        safe_neg = np.where(x < 0, -x, 0)
        features.append(-np.sqrt(safe_neg))
        names.append('sqrt_neg')
        
        # Absolute value
        features.append(np.abs(x))
        names.append('abs')
        
        # Square root of absolute
        features.append(np.sign(x) * np.sqrt(np.abs(x)))
        names.append('sign_sqrt')
        
        # Inverse
        inv_safe = np.where(np.abs(x) > 0.01, 1.0 / x, 0.0)
        features.append(inv_safe)
        names.append('inv')
        
        # x * exp(-x^2)
        features.append(x * np.exp(-x**2))
        names.append('x_exp_negx2')
        
        # Gaussian-like
        features.append(np.exp(-x**2))
        names.append('gauss_1')
        features.append(np.exp(-0.5 * x**2))
        names.append('gauss_05')
        features.append(np.exp(-2.0 * x**2))
        names.append('gauss_2')
        
        # Sigmoid-like
        features.append(1.0 / (1.0 + np.exp(-x)))
        names.append('sigmoid')
        
        # x^2 * exp(-x^2)
        features.append(x**2 * np.exp(-x**2))
        names.append('x2_exp_negx2')
        
        # Power functions
        for p in [0.5, 1.5, 2.5]:
            safe_abs = np.abs(x) + 0.01
            features.append(np.sign(x) * np.power(safe_abs, p))
            names.append(f'pow_{p}')
        
        return np.column_stack(features), names
    
    # Try feature selection approach
    try:
        if time.time() - start_time < deadline - 30:
            F, feat_names = build_features(x_norm)
            n_features = F.shape[1]
            
            # Add intercept
            F_aug = np.column_stack([np.ones(n), F])
            
            # Step 1: L1 regularization to select features
            # Use a simple iterative Lasso-like approach
            # First, normalize y
            y_mean = y_sorted.mean()
            y_std = y_sorted.std()
            if y_std < 1e-10:
                y_norm_target = np.zeros(n)
            else:
                y_norm_target = (y_sorted - y_mean) / y_std
            
            # Simple Lasso via coordinate descent
            alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0]
            best_lasso_mse = np.inf
            best_lasso_coeffs = None
            best_lasso_alpha = None
            
            for alpha in alpha_values:
                if time.time() - start_time > deadline - 20:
                    break
                try:
                    # Initialize
                    coeffs = np.zeros(F_aug.shape[1])
                    
                    # Coordinate descent for Lasso
                    for cd_iter in range(200):
                        if time.time() - start_time > deadline - 20:
                            break
                        for j in range(F_aug.shape[1]):
                            # Compute residual excluding feature j
                            residual = y_norm_target - F_aug @ coeffs + F_aug[:, j] * coeffs[j]
                            
                            # Soft threshold
                            rho = F_aug[:, j] @ residual
                            z = F_aug[:, j] @ F_aug[:, j]
                            
                            if z < 1e-10:
                                coeffs[j] = 0.0
                                continue
                                
                            if rho > alpha:
                                coeffs[j] = (rho - alpha) / z
                            elif rho < -alpha:
                                coeffs[j] = (rho + alpha) / z
                            else:
                                coeffs[j] = 0.0
                    
                    # Evaluate
                    y_pred_lasso = F_aug @ coeffs
                    mse_lasso = np.mean((y_pred_lasso - y_norm_target) ** 2)
                    
                    if mse_lasso < best_lasso_mse:
                        best_lasso_mse = mse_lasso
                        best_lasso_coeffs = coeffs.copy()
                        best_lasso_alpha = alpha
                        
                except Exception:
                    continue
            
            # Step 2: Refine with least squares on selected features
            if best_lasso_coeffs is not None:
                # Select non-zero features
                threshold = 1e-6
                selected_mask = np.abs(best_lasso_coeffs) > threshold
                if np.sum(selected_mask) > 0:
                    F_selected = F_aug[:, selected_mask]
                    
                    # Least squares on selected features
                    try:
                        coeffs_ls, residuals, rank, sv = np.linalg.lstsq(F_selected, y_norm_target, rcond=None)
                        y_pred_ls = F_selected @ coeffs_ls
                        mse_ls = np.mean((y_pred_ls - y_norm_target) ** 2)
                        
                        if mse_ls < best_lasso_mse or mse_ls < 0.01:
                            # Convert back to original scale
                            def predict_feature(X):
                                X_eval = np.asarray(X, dtype=float).flatten()
                                x_eval_norm = (X_eval - (x_min + x_max) / 2) / (x_range / 2)
                                F_eval, _ = build_features(x_eval_norm)
                                F_eval_aug = np.column_stack([np.ones(len(F_eval)), F_eval])
                                F_eval_sel = F_eval_aug[:, selected_mask]
                                y_pred_norm = F_eval_sel @ coeffs_ls
                                y_pred = y_pred_norm * y_std + y_mean
                                return y_pred.reshape(np.asarray(X).shape)
                            
                            # Check for finite values
                            test_pred = predict_feature(X_sorted.reshape(-1, 1))
                            if np.all(np.isfinite(test_pred)):
                                models.append((mse_ls * y_std**2, predict_feature))
                    except Exception:
                        pass
                
                # Also try Lasso solution directly
                y_pred_lasso_orig = F_aug @ best_lasso_coeffs
                mse_lasso_orig = np.mean((y_pred_lasso_orig - y_norm_target) ** 2) * y_std**2
                
                def predict_lasso(X):
                    X_eval = np.asarray(X, dtype=float).flatten()
                    x_eval_norm = (X_eval - (x_min + x_max) / 2) / (x_range / 2)
                    F_eval, _ = build_features(x_eval_norm)
                    F_eval_aug = np.column_stack([np.ones(len(F_eval)), F_eval])
                    y_pred_norm = F_eval_aug @ best_lasso_coeffs
                    y_pred = y_pred_norm * y_std + y_mean
                    return y_pred.reshape(np.asarray(X).shape)
                
                test_pred = predict_lasso(X_sorted.reshape(-1, 1))
                if np.all(np.isfinite(test_pred)):
                    models.append((mse_lasso_orig, predict_lasso))
    except Exception:
        pass
    
    # Model: Polynomial regression (degrees 1-12)
    best_poly_mse = np.inf
    poly_coeffs = None
    for deg in range(1, 13):
        if time.time() - start_time > deadline - 15:
            break
        try:
            coeffs = np.polyfit(X_sorted, y_sorted, deg)
            y_pred = np.polyval(coeffs, X_sorted)
            mse = np.mean((y_pred - y_sorted) ** 2)
            if mse < best_poly_mse:
                best_poly_mse = mse
                poly_coeffs = coeffs
        except Exception:
            continue
    
    if poly_coeffs is not None:
        def predict_poly(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            return np.polyval(poly_coeffs, X_eval).reshape(np.asarray(X).shape)
        models.append((best_poly_mse, predict_poly))
    
    # Model: Cubic Spline
    try:
        if len(X_sorted) >= 4:
            spline = CubicSpline(X_sorted, y_sorted)
            y_pred = spline(X_sorted)
            mse_spline = np.mean((y_pred - y_sorted) ** 2)
            def predict_spline(X):
                X_eval = np.asarray(X, dtype=float).flatten()
                return spline(X_eval).reshape(np.asarray(X).shape)
            models.append((mse_spline, predict_spline))
    except Exception:
        pass
    
    # Model: PCHIP
    try:
        if len(X_sorted) >= 3:
            pchip = PchipInterpolator(X_sorted, y_sorted)
            y_pred = pchip(X_sorted)
            mse_pchip = np.mean((y_pred - y_sorted) ** 2)
            def predict_pchip(X):
                X_eval = np.asarray(X, dtype=float).flatten()
                return pchip(X_eval).reshape(np.asarray(X).shape)
            models.append((mse_pchip, predict_pchip))
    except Exception:
        pass
    
    # Model: Specific parametric fits with good initial guesses
    if time.time() - start_time < deadline - 20:
        # Try y = a * sin(b*x + c) + d with multiple frequency guesses
        def sine_model(params, x):
            a, b, c, d = params
            return a * np.sin(b * x + c) + d
        
        y_min, y_max = y_sorted.min(), y_sorted.max()
        a0 = (y_max - y_min) / 2
        d0 = (y_max + y_min) / 2
        
        for b0 in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
            for c0 in [0.0, np.pi/4, np.pi/2, np.pi, 2*np.pi]:
                if time.time() - start_time > deadline - 15:
                    break
                try:
                    res = least_squares(sine_model, [a0, b0, c0, d0], args=(X_sorted,),
                                       method='lm', max_nfev=500)
                    y_pred = sine_model(res.x, X_sorted)
                    mse_sin = np.mean((y_pred - y_sorted) ** 2)
                    
                    if np.isfinite(mse_sin) and mse_sin < 1e-8:
                        def predict_sin(X, params=res.x):
                            X_eval = np.asarray(X, dtype=float).flatten()
                            return sine_model(params, X_eval).reshape(np.asarray(X).shape)
                        models.append((mse_sin, predict_sin))
                        break
                except Exception:
                    continue
            if time.time() - start_time > deadline - 15:
                break
    
    # Try y = a * exp(b*x) + c
    if time.time() - start_time < deadline - 15:
        try:
            def exp_model(params, x):
                a, b, c = params
                return a * np.exp(b * x) + c
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = (y_max - y_min) / 2
            c0 = (y_max + y_min) / 2
            
            for b0 in [-2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0]:
                try:
                    res = least_squares(exp_model, [a0, b0, c0], args=(X_sorted,),
                                       method='lm', max_nfev=500)
                    y_pred = exp_model(res.x, X_sorted)
                    mse_exp = np.mean((y_pred - y_sorted) ** 2)
                    
                    if np.isfinite(mse_exp) and mse_exp < 1e-8:
                        def predict_exp(X, params=res.x):
                            X_eval = np.asarray(X, dtype=float).flatten()
                            return exp_model(params, X_eval).reshape(np.asarray(X).shape)
                        models.append((mse_exp, predict_exp))
                        break
                except Exception:
                    continue
        except Exception:
            pass
    
    # Try y = a * x^b + c (for x > 0)
    if time.time() - start_time < deadline - 15:
        if np.all(X_sorted > 0.01):
            try:
                def power_model(params, x):
                    a, b, c = params
                    return a * np.power(x, b) + c
                
                y_min, y_max = y_sorted.min(), y_sorted.max()
                a0 = (y_max - y_min) / 2
                c0 = (y_max + y_min) / 2
                
                for b0 in [0.5, 1.0, 1.5, 2.0, -0.5, -1.0]:
                    try:
                        res = least_squares(power_model, [a0, b0, c0], args=(X_sorted,),
                                           method='lm', max_nfev=500)
                        y_pred = power_model(res.x, X_sorted)
                        mse_pow = np.mean((y_pred - y_sorted) ** 2)
                        
                        if np.isfinite(mse_pow) and mse_pow < 1e-8:
                            def predict_pow(X, params=res.x):
                                X_eval = np.asarray(X, dtype=float).flatten()
                                return power_model(params, X_eval).reshape(np.asarray(X).shape)
                            models.append((mse_pow, predict_pow))
                            break
                    except Exception:
                        continue
            except Exception:
                pass
    
    # Try y = a / (b + c*x) + d
    if time.time() - start_time < deadline - 15:
        try:
            def rational_model(params, x):
                a, b, c, d = params
                denom = b + c * x
                denom_safe = np.where(np.abs(denom) < 1e-6, 1e-6 * np.sign(denom + 1e-10), denom)
                return a / denom_safe + d
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = y_max - y_min
            b0 = 1.0
            c0 = 0.1
            d0 = y_min
            
            res = least_squares(rational_model, [a0, b0, c0, d0], args=(X_sorted,),
                               method='lm', max_nfev=1000)
            y_pred = rational_model(res.x, X_sorted)
            mse_rat = np.mean((y_pred - y_sorted) ** 2)
            
            if np.isfinite(mse_rat) and mse_rat < 1e-6:
                def predict_rat(X, params=res.x):
                    X_eval = np.asarray(X, dtype=float).flatten()
                    return rational_model(params, X_eval).reshape(np.asarray(X).shape)
                models.append((mse_rat, predict_rat))
        except Exception:
            pass
    
    # Try y = a * tanh(b*x) + c
    if time.time() - start_time < deadline - 15:
        try:
            def tanh_model(params, x):
                a, b, c = params
                return a * np.tanh(b * x) + c
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = (y_max - y_min) / 2
            c0 = (y_max + y_min) / 2
            
            for b0 in [0.5, 1.0, 2.0, 5.0]:
                try:
                    res = least_squares(tanh_model, [a0, b0, c0], args=(X_sorted,),
                                       method='lm', max_nfev=500)
                    y_pred = tanh_model(res.x, X_sorted)
                    mse_tanh = np.mean((y_pred - y_sorted) ** 2)
                    
                    if np.isfinite(mse_tanh) and mse_tanh < 1e-8:
                        def predict_tanh(X, params=res.x):
                            X_eval = np.asarray(X, dtype=float).flatten()
                            return tanh_model(params, X_eval).reshape(np.asarray(X).shape)
                        models.append((mse_tanh, predict_tanh))
                        break
                except Exception:
                    continue
        except Exception:
            pass
    
    # Fallback: linear interpolation
    if not models:
        def predict_interp(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            return np.interp(X_eval, X_sorted, y_sorted).reshape(np.asarray(X).shape)
        return predict_interp
    
    # Sort by MSE and pick the best
    models.sort(key=lambda x: x[0])
    
    # Check if best model is essentially perfect
    best_mse, best_predict = models[0]
    
    if best_mse < 1e-10:
        return best_predict
    
    # If best MSE is very low, use it directly
    if best_mse < 1e-6:
        # Verify on training data
        try:
            train_pred = best_predict(X_sorted.reshape(-1, 1))
            if np.all(np.isfinite(train_pred)):
                return best_predict
        except Exception:
            pass
    
    # Ensemble of top models with low MSE
    good_models = [m for m in models if m[0] < best_mse * 5 + 1e-8]
    if len(good_models) > 1:
        predict_funcs = [m[1] for m in good_models]
        def predict_ensemble(X):
            preds = []
            for pf in predict_funcs:
                try:
                    p = pf(X).flatten()
                    if np.all(np.isfinite(p)):
                        preds.append(p)
                except Exception:
                    pass
            if not preds:
                # Fallback to first model
                return predict_funcs[0](X)
            return np.mean(preds, axis=0).reshape(np.asarray(X).shape)
        return predict_ensemble
    
    return best_predict
