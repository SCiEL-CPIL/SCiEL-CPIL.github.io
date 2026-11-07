import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy.interpolate import CubicSpline, PchipInterpolator
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    X = np.asarray(X_train).flatten().astype(float)
    y = np.asarray(y_train).flatten().astype(float)
    n = len(X)
    
    # Handle constant case
    if np.var(y) < 1e-12:
        mean_y = np.mean(y)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const

    best_model = None
    best_mse = np.inf
    
    def update_best(mse, predict_func):
        nonlocal best_model, best_mse
        if mse < best_mse:
            best_mse = mse
            best_model = predict_func

    # 1. Try Polynomials (Degrees 1 to 10)
    for deg in range(1, 11):
        if time.time() > deadline - 5:
            break
        try:
            if deg > n - 1:
                break
            coef = np.polyfit(X, y, deg)
            y_pred = np.polyval(coef, X)
            mse = np.mean((y_pred - y)**2)
            
            if mse < best_mse:
                best_mse = mse
                c = coef.copy()
                def predict_poly(X, c=c):
                    x = np.asarray(X).flatten()
                    return np.polyval(c, x).reshape(np.asarray(X).shape)
                best_model = predict_poly
                
            if mse < 1e-10:
                return best_model
        except Exception:
            continue

    # 2. Rich Basis Regression (Ridge)
    # This approach builds a large design matrix of potential symbolic functions
    # and uses Ridge regression to find the best linear combination.
    # This is structurally different from just trying fixed forms.
    try:
        if time.time() < deadline - 20:
            # Define basis functions
            # Safe operations to avoid inf/nan
            x_safe = X.copy()
            
            # Basis 1: Polynomials up to degree 6
            basis_funcs = []
            for p in range(7):
                basis_funcs.append(x_safe ** p)
            
            # Basis 2: Exponentials
            # Clip x to prevent overflow in exp
            x_clip = np.clip(x_safe, -50, 50)
            basis_funcs.append(np.exp(x_clip))
            basis_funcs.append(np.exp(-x_clip))
            
            # Basis 3: Logarithms (safe)
            # log(1 + |x|) is smooth and defined everywhere
            basis_funcs.append(np.log1p(np.abs(x_safe)))
            
            # Basis 4: Hyperbolic
            basis_funcs.append(np.tanh(x_safe))
            basis_funcs.append(np.sinh(np.clip(x_safe, -20, 20)))
            basis_funcs.append(np.cosh(np.clip(x_safe, -20, 20)))
            
            # Basis 5: Trigonometric
            basis_funcs.append(np.sin(x_safe))
            basis_funcs.append(np.cos(x_safe))
            
            # Basis 6: Rational
            # 1/x is problematic at 0, use 1/(x + eps) or mask
            # Better: 1/(1+x^2) is smooth everywhere
            basis_funcs.append(1.0 / (1.0 + x_safe**2))
            
            # Build Design Matrix Phi
            # Shape (n, num_features)
            Phi = np.column_stack(basis_funcs)
            
            # Check for non-finite values in Phi
            if not np.all(np.isfinite(Phi)):
                # Replace non-finite with 0 or large value, or skip this method
                # For Ridge, we can just set them to 0
                Phi = np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Try different regularization strengths (lambda)
            # lambda = 0 is OLS, large lambda shrinks coefficients
            lambdas = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]
            
            for lam in lambdas:
                if time.time() > deadline - 5:
                    break
                try:
                    # Ridge Regression: (Phi^T Phi + lam * I)^-1 Phi^T y
                    # Use solve for stability
                    A = Phi.T @ Phi
                    b = Phi.T @ y
                    I = np.eye(Phi.shape[1])
                    A_reg = A + lam * I
                    
                    # If lam is 0, A might be singular, use lstsq
                    if lam == 0:
                        w, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
                    else:
                        w = np.linalg.solve(A_reg, b)
                    
                    y_pred = Phi @ w
                    
                    if not np.all(np.isfinite(y_pred)):
                        continue
                        
                    mse = np.mean((y_pred - y)**2)
                    
                    if mse < best_mse:
                        best_mse = mse
                        w_copy = w.copy()
                        # We need to store the basis definition or re-compute it in predict
                        # Since basis depends on X, we can't just store w.
                        # We must reconstruct Phi in predict.
                        # This is inefficient but acceptable for 200 points.
                        # Optimization: Store the basis functions? No, they depend on X.
                        # We will just re-evaluate the basis in predict.
                        
                        def predict_ridge(X, w=w_copy, basis_funcs=basis_funcs):
                            x = np.asarray(X).flatten().astype(float)
                            x_safe = x.copy()
                            x_clip = np.clip(x_safe, -50, 50)
                            
                            # Reconstruct basis
                            bf = []
                            for p in range(7):
                                bf.append(x_safe ** p)
                            bf.append(np.exp(x_clip))
                            bf.append(np.exp(-x_clip))
                            bf.append(np.log1p(np.abs(x_safe)))
                            bf.append(np.tanh(x_safe))
                            bf.append(np.sinh(np.clip(x_safe, -20, 20)))
                            bf.append(np.cosh(np.clip(x_safe, -20, 20)))
                            bf.append(np.sin(x_safe))
                            bf.append(np.cos(x_safe))
                            bf.append(1.0 / (1.0 + x_safe**2))
                            
                            Phi_new = np.column_stack(bf)
                            Phi_new = np.nan_to_num(Phi_new, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            pred = Phi_new @ w
                            return pred.reshape(np.asarray(X).shape)
                        
                        best_model = predict_ridge
                        
                    if mse < 1e-10:
                        return best_model
                        
                except Exception:
                    continue
    except Exception:
        pass

    # 3. Parametric Curve Fits for specific common forms
    # This helps if the function is exactly one of these but the linear basis didn't capture it well due to scaling.
    def try_curve_fit(func, p0_list, bounds=None):
        nonlocal best_model, best_mse
        for p0 in p0_list:
            if time.time() > deadline - 5:
                break
            try:
                res = least_squares(lambda p: func(X, *p) - y, p0, bounds=bounds, max_nfev=2000)
                y_pred = func(X, *res.x)
                if not np.all(np.isfinite(y_pred)):
                    continue
                mse = np.mean((y_pred - y)**2)
                if mse < best_mse:
                    best_mse = mse
                    p_opt = res.x.copy()
                    def predict_cf(X, p=p_opt, f=func):
                        x = np.asarray(X).flatten()
                        return f(x, *p).reshape(np.asarray(X).shape)
                    best_model = predict_cf
                if mse < 1e-10:
                    return True
            except Exception:
                continue
        return False

    # Form: a * exp(bx) + c
    def f_exp(x, a, b, c):
        return a * np.exp(np.clip(b*x, -50, 50)) + c
    p0_exp = [
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [np.mean(y), 0.1, 0], [np.mean(y), -0.1, 0]
    ]
    if not try_curve_fit(f_exp, p0_exp):
        pass

    # Form: a * tanh(bx) + c
    def f_tanh(x, a, b, c):
        return a * np.tanh(b*x) + c
    p0_tanh = [
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [np.mean(y), 0.1, 0]
    ]
    if not try_curve_fit(f_tanh, p0_tanh):
        pass

    # Form: a * log(1 + |bx|) + c  (Safe log)
    def f_log(x, a, b, c):
        return a * np.log1p(np.abs(b*x)) + c
    p0_log = [
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [np.mean(y), 0.1, 0]
    ]
    if not try_curve_fit(f_log, p0_log):
        pass

    # Form: a / (1 + exp(-bx)) + c (Logistic)
    def f_logistic(x, a, b, c):
        z = np.clip(-b*x, -50, 50)
        return a / (1 + np.exp(z)) + c
    p0_logistic = [
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [np.mean(y), 0.1, 0]
    ]
    if not try_curve_fit(f_logistic, p0_logistic):
        pass

    # 4. Interpolation Fallbacks
    try:
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        # Handle duplicates
        unique_X, inverse_indices = np.unique(X_sorted, return_inverse=True)
        if len(unique_X) < n:
            unique_y = np.zeros(len(unique_X))
            counts = np.zeros(len(unique_X))
            np.add.at(unique_y, inverse_indices, y_sorted)
            np.add.at(counts, inverse_indices, 1)
            unique_y /= counts
            X_sorted = unique_X
            y_sorted = unique_y
            
        # Try Cubic Spline
        cs = CubicSpline(X_sorted, y_sorted, bc_type='natural')
        y_pred = cs(X)
        mse = np.mean((y_pred - y)**2)
        
        if mse < best_mse:
            best_mse = mse
            cs_copy = cs
            def predict_spline(X):
                x = np.asarray(X).flatten()
                return cs_copy(x).reshape(np.asarray(X).shape)
            best_model = predict_spline
            
        # Try PCHIP (Monotone Cubic)
        if time.time() < deadline - 5:
            pchip = PchipInterpolator(X_sorted, y_sorted)
            y_pred = pchip(X)
            mse = np.mean((y_pred - y)**2)
            
            if mse < best_mse:
                best_mse = mse
                pchip_copy = pchip
                def predict_pchip(X):
                    x = np.asarray(X).flatten()
                    return pchip_copy(x).reshape(np.asarray(X).shape)
                best_model = predict_pchip
            
    except Exception:
        pass

    # 5. Fallback: Mean
    if best_model is None:
        mean_y = np.mean(y)
        def predict_mean(X):
            return np.full(np.asarray(X).shape, mean_y)
        best_model = predict_mean

    return best_model
