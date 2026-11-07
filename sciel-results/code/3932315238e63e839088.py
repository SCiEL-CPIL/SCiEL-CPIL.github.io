import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import minimize
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # Safety margin from 330s limit
    
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] for kernel stability
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    # Center y
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    
    # Split for validation
    n_val = max(10, int(n * 0.2))
    val_idx = np.arange(n - n_val, n)
    train_idx = np.arange(0, n - n_val)
    
    X_tr_norm = X_norm[train_idx]
    y_tr = y_centered[train_idx]
    X_val_norm = X_norm[val_idx]
    y_val = y_centered[val_idx]
    
    # ============ MODEL 1: Polynomial Kernel Ridge Regression ============
    def poly_kernel(X1, X2, degree, gamma, coef0):
        """Polynomial kernel: (gamma * X1.X2 + coef0)^degree"""
        K = (gamma * (X1 @ X2.T) + coef0) ** degree
        if len(X1) == len(X2):
            K += 1e-8 * np.eye(len(X1))
        return K
    
    def loo_cv_poly_kernel(X, y, degree, gamma, coef0, alpha):
        """Leave-one-out CV for polynomial kernel ridge."""
        n = X.shape[0]
        K = poly_kernel(X, X, degree, gamma, coef0)
        A = K + alpha * np.eye(n)
        try:
            L = np.linalg.cholesky(A)
            # LOO predictions: y_i_pred = y_i - K_ii / (L_ii^2 * (1 + K_ii/alpha))
            # Actually for kernel ridge: f_i = y_i - K_ii / (alpha + K_ii) * (y_i - f_i)
            # Simpler: use the formula alpha_i = K_ii / (alpha + K_ii)
            diag = np.diag(K)
            alpha_loo = diag / (alpha + diag)
            y_pred_loo = y - alpha_loo * y
            # More accurate: solve for each
            # For small n, do explicit LOO
            mse_sum = 0.0
            for i in range(min(n, 50)):  # Limit for speed
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                K_train = poly_kernel(X[mask], X[mask], degree, gamma, coef0)
                A_train = K_train + alpha * np.eye(n - 1)
                try:
                    w = solve(A_train, y[mask], assume_a='pos')
                    K_test = poly_kernel(X[i:i+1], X[mask], degree, gamma, coef0)
                    y_pred = K_test @ w
                    mse_sum += (y_pred[0] - y[i]) ** 2
                except:
                    mse_sum += 1e10
            return mse_sum / min(n, 50)
        except:
            return 1e10
    
    # ============ MODEL 2: Matern-5/2 GP ============
    def matern52_kernel(X1, X2, length_scale, signal_var):
        """Matern 5/2 kernel."""
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        r = np.sqrt(sq_dists)
        K = signal_var * (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r / length_scale)
        if len(X1) == len(X2):
            K += 1e-8 * np.eye(len(X1))
        return K
    
    def neg_log_likelihood_matern(params):
        """NLL for Matern-5/2 GP."""
        length_scale = np.exp(params[0])
        signal_var = np.exp(params[1])
        noise_var = np.exp(params[2])
        
        K = matern52_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            nll = 0.5 * y_tr @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y_tr) * np.log(2 * np.pi)
            return nll
        except:
            return 1e10
    
    # ============ MODEL 3: Rational Function Approximator ============
    def rational_features(X, num_degree, den_degree):
        """Generate features for rational function: numerator/denominator."""
        n = X.shape[0]
        d = X.shape[1]
        
        # Numerator: polynomial up to num_degree
        num_feats = [np.ones(n)]
        for deg in range(1, num_degree + 1):
            for i in range(d):
                for j in range(i, d):
                    # All combinations with repetition of degree deg
                    for combo in _combinations_with_replacement(range(d), deg):
                        term = np.ones(n)
                        for idx in combo:
                            term *= X[:, idx]
                        num_feats.append(term)
        
        # Denominator: polynomial up to den_degree (offset by 1 to avoid zero)
        den_feats = [np.ones(n) + 0.1]  # Small offset for stability
        for deg in range(1, den_degree + 1):
            for i in range(d):
                for j in range(i, d):
                    for combo in _combinations_with_replacement(range(d), deg):
                        term = np.ones(n)
                        for idx in combo:
                            term *= X[:, idx]
                        den_feats.append(term)
        
        return np.column_stack(num_feats), np.column_stack(den_feats)
    
    def _combinations_with_replacement(iterable, r):
        """Generate combinations with replacement."""
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return
            indices[i:] = range(indices[i] + 1, indices[i] + 1 + r - i)
            yield tuple(pool[i] for i in indices)
    
    def fit_rational(X, y, num_degree, den_degree, alpha):
        """Fit rational function using linear least squares on numerator/denominator."""
        num_feat, den_feat = rational_features(X, num_degree, den_degree)
        
        # We want to find w_num, w_den such that y ≈ num_feat @ w_num / (den_feat @ w_den)
        # Linearize: y * (den_feat @ w_den) ≈ num_feat @ w_num
        # This is still nonlinear. Use iterative approach or linearize differently.
        # Alternative: fit y * den_feat @ w_den = num_feat @ w_num
        # Stack: [num_feat, -y*den_feat] @ [w_num; w_den] = 0
        # This is underdetermined. Use different approach.
        
        # Simpler: assume denominator is a fixed simple function, fit numerator
        # Or: use the fact that for rational f(x) = N(x)/D(x), we can do:
        # y * D(x) = N(x)
        # If we fix D(x) = 1 + sum of simple terms, we can linearize
        
        # Let's use a simpler rational model: y = (a0 + a1*x1 + ... + am*xm) / (1 + b1*x1 + ... + bm*xm)
        # Linearize: y * (1 + b1*x1 + ... + bm*xm) = a0 + a1*x1 + ... + am*xm
        # y + y*b1*x1 + ... + y*bm*xm = a0 + a1*x1 + ... + am*xm
        # y - a0 - a1*x1 - ... - am*xm = -y*b1*x1 - ... - y*bm*xm
        # This is still coupled. Use iterative reweighted least squares.
        
        # For simplicity, use a linear denominator: D(x) = 1 + c1*x1 + ... + cd*xd
        # Then: y * (1 + c1*x1 + ... + cd*xd) = N(x)
        # y + y*c1*x1 + ... + y*cd*xd = N(x)
        # N(x) - y = y*c1*x1 + ... + y*cd*xd
        # This is linear in c if N is known, or linear in N coefficients if c is known.
        
        # Use alternating minimization
        n_samples = X.shape[0]
        d_input = X.shape[1]
        
        # Initialize denominator coefficients
        c = np.zeros(d_input)
        w_num = np.zeros(num_feat.shape[1])
        
        for iteration in range(10):
            # Given c, solve for w_num: N(x) = y * D(x)
            D = 1.0 + X @ c
            target = y * D
            A_num = num_feat.T @ num_feat + alpha * np.eye(num_feat.shape[1])
            b_num = num_feat.T @ target
            try:
                w_num = solve(A_num, b_num, assume_a='pos')
            except:
                w_num = np.linalg.lstsq(num_feat, target, rcond=None)[0]
            
            # Given w_num, solve for c: y * D(x) = N(x) => D(x) = N(x)/y
            N_pred = num_feat @ w_num
            valid = np.abs(y) > 1e-6
            if np.sum(valid) < d_input + 1:
                break
            target_D = N_pred[valid] / y[valid]
            # D(x) = 1 + X @ c, so target_D - 1 = X[valid] @ c
            X_valid = X[valid]
            A_c = X_valid.T @ X_valid + alpha * np.eye(d_input)
            b_c = X_valid.T @ (target_D - 1.0)
            try:
                c = solve(A_c, b_c, assume_a='pos')
            except:
                c = np.linalg.lstsq(X_valid, target_D - 1.0, rcond=None)[0]
        
        # Predict function
        def predict_rational(X_test):
            num_feat_test, den_feat_test = rational_features(X_test, num_degree, den_degree)
            N = num_feat_test @ w_num
            D = 1.0 + X_test @ c
            D = np.maximum(np.abs(D), 1e-6)  # Avoid division by zero
            return N / D
        
        # Compute LOO CV score
        mse_sum = 0.0
        n_check = min(n_samples, 30)
        for i in range(n_check):
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False
            try:
                c_i, w_num_i = _fit_rational_inner(X[mask], y[mask], num_feat[:, mask], num_degree, den_degree, alpha)
                num_feat_i, _ = rational_features(X[i:i+1], num_degree, den_degree)
                N_i = num_feat_i @ w_num_i
                D_i = 1.0 + X[i] @ c_i
                D_i = max(abs(D_i), 1e-6)
                y_pred_i = N_i[0] / D_i
                mse_sum += (y_pred_i - y[i]) ** 2
            except:
                mse_sum += 1e10
        return mse_sum / n_check, predict_rational, (c, w_num)
    
    def _fit_rational_inner(X, y, num_feat, num_degree, den_degree, alpha):
        """Inner fit for rational model (no LOO)."""
        d_input = X.shape[1]
        c = np.zeros(d_input)
        w_num = np.zeros(num_feat.shape[1])
        
        for iteration in range(10):
            D = 1.0 + X @ c
            target = y * D
            A_num = num_feat.T @ num_feat + alpha * np.eye(num_feat.shape[1])
            b_num = num_feat.T @ target
            try:
                w_num = solve(A_num, b_num, assume_a='pos')
            except:
                w_num = np.linalg.lstsq(num_feat, target, rcond=None)[0]
            
            N_pred = num_feat @ w_num
            valid = np.abs(y) > 1e-6
            if np.sum(valid) < d_input + 1:
                break
            target_D = N_pred[valid] / y[valid]
            X_valid = X[valid]
            A_c = X_valid.T @ X_valid + alpha * np.eye(d_input)
            b_c = X_valid.T @ (target_D - 1.0)
            try:
                c = solve(A_c, b_c, assume_a='pos')
            except:
                c = np.linalg.lstsq(X_valid, target_D - 1.0, rcond=None)[0]
        
        return c, w_num
    
    # ============ Fit and evaluate all models ============
    model_scores = {}
    model_predictors = {}
    
    # --- Fit Polynomial Kernel Ridge ---
    best_poly_score = np.inf
    best_poly_params = None
    
    if time.time() < deadline - 80:
        for degree in [2, 3, 4]:
            for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
                for coef0 in [0.1, 1.0, 10.0]:
                    for alpha in [0.01, 0.1, 1.0, 10.0]:
                        if time.time() > deadline - 60:
                            break
                        score = loo_cv_poly_kernel(X_tr_norm, y_tr, degree, gamma, coef0, alpha)
                        if score < best_poly_score:
                            best_poly_score = score
                            best_poly_params = (degree, gamma, coef0, alpha)
    
    if best_poly_params is not None:
        degree, gamma, coef0, alpha = best_poly_params
        K_full = poly_kernel(X_norm, X_norm, degree, gamma, coef0)
        A_full = K_full + alpha * np.eye(n)
        try:
            w_full = solve(A_full, y_centered, assume_a='pos')
            def predict_poly(X):
                X_test_norm = (X - min_vals) / ranges
                K_test = poly_kernel(X_test_norm, X_norm, degree, gamma, coef0)
                return K_test @ w_full + y_mean
            model_scores['poly'] = best_poly_score
            model_predictors['poly'] = predict_poly
        except:
            pass
    
    # --- Fit Matern-5/2 GP ---
    best_matern_score = np.inf
    best_matern_params = None
    
    if time.time() < deadline - 60:
        # Grid search
        for ls in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            for sv in [0.1, 1.0, 10.0]:
                for nv in [0.001, 0.01, 0.1]:
                    params = np.array([np.log(ls), np.log(sv), np.log(nv)])
                    nll = neg_log_likelihood_matern(params)
                    if nll < best_matern_score:
                        best_matern_score = nll
                        best_matern_params = params
        
        # Refine
        if best_matern_params is not None:
            try:
                result = minimize(neg_log_likelihood_matern, best_matern_params, method='L-BFGS-B',
                                 bounds=[(-4, 4), (-2, 4), (-8, -1)], maxiter=50)
                if result.fun < best_matern_score:
                    best_matern_score = result.fun
                    best_matern_params = result.x
            except:
                pass
    
    if best_matern_params is not None:
        length_scale = np.exp(best_matern_params[0])
        signal_var = np.exp(best_matern_params[1])
        noise_var = np.exp(best_matern_params[2])
        
        # Validate on holdout
        K_val_tr = matern52_kernel(X_val_norm, X_tr_norm, length_scale, signal_var)
        K_tr_tr = matern52_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
        try:
            w = solve(K_tr_tr, y_tr, assume_a='pos')
            y_pred_val = K_val_tr @ w
            val_mse_matern = np.mean((y_pred_val - y_val) ** 2)
            
            # Refit on all data
            K_full = matern52_kernel(X_norm, X_norm, length_scale, signal_var) + noise_var * np.eye(n)
            w_full = solve(K_full, y_centered, assume_a='pos')
            
            def predict_matern(X):
                X_test_norm = (X - min_vals) / ranges
                K_test = matern52_kernel(X_test_norm, X_norm, length_scale, signal_var)
                return K_test @ w_full + y_mean
            
            model_scores['matern'] = val_mse_matern
            model_predictors['matern'] = predict_matern
        except:
            pass
    
    # --- Fit Rational Function Approximator ---
    best_rat_score = np.inf
    best_rat_predict = None
    
    if time.time() < deadline - 40:
        for num_deg in [2, 3]:
            for den_deg in [1, 2]:
                for alpha in [0.01, 0.1, 1.0]:
                    if time.time() > deadline - 20:
                        break
                    try:
                        score, pred_func, _ = fit_rational(X_tr_norm, y_tr, num_deg, den_deg, alpha)
                        if score < best_rat_score:
                            best_rat_score = score
                            # Refit on all data
                            c_full, w_num_full = _fit_rational_inner(X_norm, y_centered, 
                                rational_features(X_norm, num_deg, den_deg)[0], num_deg, den_deg, alpha)
                            def predict_rational_full(X, c=c_full, w_num=w_num_full, nd=num_deg, dd=den_deg):
                                X_test_norm = (X - min_vals) / ranges
                                num_feat_test, _ = rational_features(X_test_norm, nd, dd)
                                N = num_feat_test @ w_num
                                D = 1.0 + X_test_norm @ c
                                D = np.maximum(np.abs(D), 1e-6)
                                return N / D + y_mean
                            best_rat_predict = predict_rational_full
                    except:
                        continue
    
    if best_rat_predict is not None:
        model_scores['rational'] = best_rat_score
        model_predictors['rational'] = best_rat_predict
    
    # ============ Select best model ============
    if not model_scores:
        # Fallback: simple linear regression
        X_design = np.hstack([np.ones((n, 1)), X_norm])
        w = np.linalg.lstsq(X_design, y_centered, rcond=None)[0]
        def predict_fallback(X):
            X_test_norm = (X - min_vals) / ranges
            X_design_test = np.hstack([np.ones((X.shape[0], 1)), X_test_norm])
            return X_design_test @ w + y_mean
        return predict_fallback
    
    # Select model with lowest validation score
    best_model_name = min(model_scores, key=model_scores.get)
    return model_predictors[best_model_name]
