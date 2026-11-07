import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import cho_factor, cho_solve
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit (330s kill)
    
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

    # 1. Try Polynomials (Degrees 1 to 15) with Ridge Regression for stability
    # Actually, simple polyfit is fine for low degrees, but Ridge is safer for high.
    for deg in range(1, 16):
        if time.time() > deadline - 10:
            break
        try:
            if deg > n - 1:
                break
            # Use np.vander for design matrix
            # np.polyfit uses SVD, which is stable.
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

    # 2. Systematic Basis Expansion with Ridge Regression
    # Features: Polynomials, Sines, Cosines, Exponentials, Rational approximations
    # We build a matrix Phi where columns are basis functions.
    # Then solve (Phi^T Phi + lambda I) w = Phi^T y
    
    def build_basis(x):
        feats = []
        # Polynomials up to degree 5
        for p in range(1, 6):
            feats.append(x**p)
        # Trig functions (frequencies 1, 2, 3)
        for k in [1, 2, 3]:
            feats.append(np.sin(k * x))
            feats.append(np.cos(k * x))
        # Exponentials (decay and growth, centered)
        # Normalize x to roughly [-1, 1] for stable exp?
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-6)
        feats.append(np.exp(x_norm))
        feats.append(np.exp(-x_norm))
        # Gaussian bump
        feats.append(np.exp(-x_norm**2))
        # Rational-like: 1/(1+x^2)
        feats.append(1.0 / (1.0 + x**2))
        
        return np.column_stack(feats)
    
    try:
        if time.time() < deadline - 20:
            Phi = build_basis(X)
            # Add intercept
            Phi = np.column_stack([np.ones(n), Phi])
            
            # Try a few regularization strengths
            for lam in [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]:
                A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
                b = Phi.T @ y
                try:
                    w = np.linalg.solve(A, b)
                    y_pred = Phi @ w
                    mse = np.mean((y_pred - y)**2)
                    
                    if mse < best_mse:
                        best_mse = mse
                        w_copy = w.copy()
                        def predict_basis(X, w=w_copy):
                            x = np.asarray(X).flatten()
                            Phi_test = build_basis(x)
                            Phi_test = np.column_stack([np.ones(len(x)), Phi_test])
                            return (Phi_test @ w).reshape(np.asarray(X).shape)
                        best_model = predict_basis
                        
                    if mse < 1e-10:
                        return best_model
                except np.linalg.LinAlgError:
                    continue
    except Exception:
        pass

    # 3. Gaussian Process Regression (Manual Implementation)
    # K(x, x') = sigma_f^2 * exp(-0.5 * (x - x')^2 / l^2)
    # We will try a grid of (l, sigma_f)
    try:
        if time.time() < deadline - 30:
            # Normalize inputs to [0, 1] for kernel stability? Or use raw.
            # Raw is fine if we pick l appropriately.
            # Let's estimate scale from data
            x_scale = np.max(X) - np.min(X)
            if x_scale < 1e-6:
                x_scale = 1.0
                
            # Grid search for hyperparameters
            # l (length scale): relative to x_scale
            # sigma_f (signal variance): relative to y variance
            y_var = np.var(y)
            if y_var < 1e-6:
                y_var = 1.0
                
            l_options = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
            sf_options = [0.5, 1.0, 2.0, 5.0]
            
            for l in l_options:
                if time.time() > deadline - 10:
                    break
                l_val = l * x_scale
                for sf in sf_options:
                    if time.time() > deadline - 10:
                        break
                    sf_val = sf * y_var
                    
                    # Kernel matrix K
                    # K[i, j] = sf_val * exp(-0.5 * (X[i] - X[j])^2 / l_val^2)
                    diff = X[:, None] - X[None, :]
                    K = sf_val * np.exp(-0.5 * diff**2 / l_val**2)
                    
                    # Add noise term for stability (small)
                    K += 1e-6 * np.eye(n)
                    
                    try:
                        # Cholesky decomposition for stable solve
                        L = np.linalg.cholesky(K)
                        # Solve K alpha = y => alpha = L^{-T} L^{-1} y
                        alpha = cho_solve((L, True), y)
                        
                        # Predict on training data to get MSE
                        y_pred = K @ alpha
                        mse = np.mean((y_pred - y)**2)
                        
                        if mse < best_mse:
                            best_mse = mse
                            # Store parameters for prediction
                            K_train = K
                            alpha_vec = alpha
                            l_store = l_val
                            sf_store = sf_val
                            
                            def predict_gp(X, K_train=K_train, alpha_vec=alpha_vec, l=l_store, sf=sf_store):
                                x = np.asarray(X).flatten()
                                # Kernel between test and train
                                diff_test = x[:, None] - X[None, :]
                                K_test = sf * np.exp(-0.5 * diff_test**2 / l**2)
                                return (K_test @ alpha_vec).reshape(np.asarray(X).shape)
                            best_model = predict_gp
                            
                        if mse < 1e-10:
                            return best_model
                    except np.linalg.LinAlgError:
                        continue
    except Exception:
        pass

    # 4. Interpolation Fallbacks (Spline, PCHIP)
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
