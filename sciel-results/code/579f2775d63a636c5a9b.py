import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import solve, cho_factor, cho_solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 300.0  # 300s limit, keep 30s margin
    
    X = X_train.reshape(-1, 1)
    y = y_train.reshape(-1)
    n_samples = X.shape[0]
    
    # Normalize X to [-1, 1] for better kernel behavior
    x_min = X.min()
    x_max = X.max()
    x_range = x_max - x_min
    if x_range == 0:
        X_norm = np.zeros_like(X)
    else:
        X_norm = (X - x_min) / x_range * 2 - 1
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    candidates = []
    
    # Helper to evaluate a model's NMSE on validation set
    def evaluate_model(predict_fn, X_val, y_val):
        try:
            y_pred = predict_fn(X_val)
            mse = np.mean((y_pred - y_val) ** 2)
            var_y = np.var(y_val)
            if var_y < 1e-10:
                return 0.0 if mse < 1e-10 else float('inf')
            return mse / var_y
        except:
            return float('inf')

    # K-Fold Cross Validation
    k = 5
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    def cv_score(model_factory):
        """
        model_factory: function that takes train indices and returns a predict function for full space
        """
        scores = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            X_tr = X_norm[train_idx]
            y_tr = y_norm[train_idx]
            X_val = X_norm[val_idx]
            y_val = y_norm[val_idx]
            
            # Train model on train fold
            try:
                predict_fn = model_factory(X_tr, y_tr)
                # Evaluate on val fold
                score = evaluate_model(predict_fn, X_val, y_val)
                scores.append(score)
            except:
                scores.append(float('inf'))
        
        if any(np.isinf(s) for s in scores):
            return float('inf')
        return np.mean(scores)

    # --- Model 1: Gaussian Process with RBF Kernel ---
    def fit_gpr_rbf(X_tr, y_tr):
        def kernel(X1, X2, l, s):
            # RBF kernel
            dist_sq = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            dist_sq = np.maximum(dist_sq, 0)
            return s**2 * np.exp(-0.5 * dist_sq / (l**2))
        
        def neg_lml(params):
            l, s, noise = params
            l = np.exp(l)
            s = np.exp(s)
            noise = np.exp(noise)
            K = kernel(X_tr, X_tr, l, s) + noise * np.eye(len(X_tr))
            try:
                L, info = cho_factor(K)
                if info > 0:
                    return 1e10
                logdet = 2 * np.sum(np.log(np.diag(cho_factor(K)[0]))) # Re-factor for logdet? No, use slogdet
                # Better way:
                # L = cho_factor(K, lower=True)[0]
                # logdet = 2 * sum(log(diag(L)))
                alpha = cho_solve((L, True), y_tr)
                lml = -0.5 * y_tr @ alpha - 0.5 * np.log(np.linalg.det(K)) - 0.5 * len(X_tr) * np.log(2 * np.pi)
                return -lml
            except:
                return 1e10
        
        # Initial guess
        x0 = [0.0, 0.0, -2.0] # log(l), log(s), log(noise)
        res = minimize(neg_lml, x0, method='L-BFGS-B', bounds=[(-5, 5), (-5, 5), (-10, 0)])
        l, s, noise = np.exp(res.x[0]), np.exp(res.x[1]), np.exp(res.x[2])
        
        K = kernel(X_tr, X_tr, l, s) + noise * np.eye(len(X_tr))
        L, info = cho_factor(K)
        if info > 0:
            # Fallback to simple linear
            def predict_linear(X_new):
                w, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X_tr)), X_tr]), y_tr, rcond=None)
                return np.column_stack([np.ones(len(X_new)), X_new]) @ w
            return predict_linear
        
        alpha = cho_solve((L, True), y_tr)
        
        def predict_gpr(X_new):
            K_star = kernel(X_new, X_tr, l, s)
            return K_star @ alpha
        
        return predict_gpr

    # --- Model 2: Gaussian Process with Matern 5/2 Kernel ---
    def fit_gpr_matern(X_tr, y_tr):
        def kernel(X1, X2, l, s):
            # Matern 5/2
            dist = np.sqrt(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T)
            dist = np.maximum(dist, 1e-10)
            sqrt5 = np.sqrt(5)
            return s**2 * (1 + sqrt5 * dist / l + 5 * dist**2 / (3 * l**2)) * np.exp(-sqrt5 * dist / l)
        
        def neg_lml(params):
            l, s, noise = params
            l = np.exp(l)
            s = np.exp(s)
            noise = np.exp(noise)
            K = kernel(X_tr, X_tr, l, s) + noise * np.eye(len(X_tr))
            try:
                L, info = cho_factor(K)
                if info > 0:
                    return 1e10
                alpha = cho_solve((L, True), y_tr)
                logdet = 2 * np.sum(np.log(np.diag(L)))
                lml = -0.5 * y_tr @ alpha - 0.5 * logdet - 0.5 * len(X_tr) * np.log(2 * np.pi)
                return -lml
            except:
                return 1e10
        
        x0 = [0.0, 0.0, -2.0]
        res = minimize(neg_lml, x0, method='L-BFGS-B', bounds=[(-5, 5), (-5, 5), (-10, 0)])
        l, s, noise = np.exp(res.x[0]), np.exp(res.x[1]), np.exp(res.x[2])
        
        K = kernel(X_tr, X_tr, l, s) + noise * np.eye(len(X_tr))
        L, info = cho_factor(K)
        if info > 0:
            def predict_linear(X_new):
                w, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X_tr)), X_tr]), y_tr, rcond=None)
                return np.column_stack([np.ones(len(X_new)), X_new]) @ w
            return predict_linear
        
        alpha = cho_solve((L, True), y_tr)
        
        def predict_gpr(X_new):
            K_star = kernel(X_new, X_tr, l, s)
            return K_star @ alpha
        
        return predict_gpr

    # --- Model 3: Ridge Polynomial ---
    def make_poly_model(degree, alpha):
        def fit_poly(X_tr, y_tr):
            # Design matrix
            X_pow = np.polynomial.polynomial.polyvander(X_tr, degree)
            # Ridge regression
            # Minimize ||X w - y||^2 + alpha ||w||^2
            # (X^T X + alpha I) w = X^T y
            A = X_pow.T @ X_pow + alpha * np.eye(degree + 1)
            b = X_pow.T @ y_tr
            try:
                w = np.linalg.solve(A, b)
            except:
                w, *_ = np.linalg.lstsq(X_pow, y_tr, rcond=None)
            
            def predict_poly(X_new):
                X_new_pow = np.polynomial.polynomial.polyvander(X_new, degree)
                return X_new_pow @ w
            return predict_poly
        return fit_poly

    # --- Model 4: Fourier Series ---
    def make_fourier_model(K, alpha):
        def fit_fourier(X_tr, y_tr):
            # Basis: 1, sin(x), cos(x), ..., sin(Kx), cos(Kx)
            cols = [np.ones(len(X_tr))]
            for k in range(1, K + 1):
                cols.append(np.sin(k * X_tr[:, 0]))
                cols.append(np.cos(k * X_tr[:, 0]))
            X_basis = np.column_stack(cols)
            
            A = X_basis.T @ X_basis + alpha * np.eye(len(cols))
            b = X_basis.T @ y_tr
            try:
                w = np.linalg.solve(A, b)
            except:
                w, *_ = np.linalg.lstsq(X_basis, y_tr, rcond=None)
            
            def predict_fourier(X_new):
                cols_new = [np.ones(len(X_new))]
                for k in range(1, K + 1):
                    cols_new.append(np.sin(k * X_new[:, 0]))
                    cols_new.append(np.cos(k * X_new[:, 0]))
                X_new_basis = np.column_stack(cols_new)
                return X_new_basis @ w
            return predict_fourier
        return fit_fourier

    # Define candidate configurations
    # Each candidate is a factory function for CV
    candidate_configs = [
        ("GPR_RBF", fit_gpr_rbf),
        ("GPR_Matern", fit_gpr_matern),
        ("Poly_3", make_poly_model(3, 1e-5)),
        ("Poly_5", make_poly_model(5, 1e-5)),
        ("Poly_7", make_poly_model(7, 1e-4)),
        ("Poly_9", make_poly_model(9, 1e-3)),
        ("Fourier_5", make_fourier_model(5, 1e-5)),
        ("Fourier_10", make_fourier_model(10, 1e-4)),
        ("Fourier_20", make_fourier_model(20, 1e-3)),
    ]

    best_score = float('inf')
    best_model_factory = None
    best_name = None

    for name, factory in candidate_configs:
        if time.time() > deadline - 30:
            break
        
        # Run CV
        try:
            score = cv_score(factory)
            if score < best_score:
                best_score = score
                best_model_factory = factory
                best_name = name
        except Exception as e:
            pass

    # If no model worked, fallback to linear
    if best_model_factory is None:
        def fit_linear(X_tr, y_tr):
            A = np.column_stack([np.ones(len(X_tr)), X_tr])
            w, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
            def predict(X_new):
                A_new = np.column_stack([np.ones(len(X_new)), X_new])
                return A_new @ w
            return predict
        best_model_factory = fit_linear
        best_name = "Linear_Fallback"

    # Retrain best model on full training set
    final_predict_norm = best_model_factory(X_norm, y_norm)
    
    # Wrap to handle denormalization
    def predict(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Denormalize X
        if x_range == 0:
            X_norm = np.zeros_like(X)
        else:
            X_norm = (X - x_min) / x_range * 2 - 1
            
        y_pred_norm = final_predict_norm(X_norm)
        
        # Denormalize y
        y_pred = y_pred_norm * y_std + y_mean
        
        # Ensure finite
        y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)
        
        return y_pred

    return predict
