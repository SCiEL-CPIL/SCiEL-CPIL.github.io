import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 325.0  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize X to [0, 1] or mean 0, std 1 for better kernel behavior
    # Using mean/std is standard for GPR
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    X_std = (X - mu) / sd
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_std_normalized = (y - y_mean) / y_std
    
    def make_predictor_gpr(X_std, y_std, kernel_params):
        """
        Creates a predictor function using GPR.
        kernel_params: dict with 'l' (length scale), 'sigma_f' (signal variance), 'sigma_n' (noise variance)
        """
        l = kernel_params['l']
        sigma_f = kernel_params['sigma_f']
        sigma_n = kernel_params['sigma_n']
        
        # Precompute K and alpha for training data
        # K(x, x') = sigma_f^2 * (1 + sqrt(5)*d/sqrt(5)) * exp(-sqrt(5)*d/sqrt(5)) for Matern 5/2
        # d is Euclidean distance
        
        def matern52_kernel(X1, X2, l, sigma_f):
            # Compute squared distances
            # Using cdist is efficient
            dists = cdist(X1, X2, metric='euclidean')
            # Matern 5/2
            # k(r) = (1 + sqrt(5)r + 5r^2/3) * exp(-sqrt(5)r)
            # Note: r here is scaled by 1/l
            r = dists / l
            term1 = 1 + np.sqrt(5) * r + (5/3) * r**2
            term2 = np.exp(-np.sqrt(5) * r)
            return sigma_f**2 * term1 * term2
        
        K = matern52_kernel(X_std, X_std, l, sigma_f)
        K += sigma_n**2 * np.eye(n)
        
        # Solve K alpha = y
        try:
            # Use Cholesky for stability if possible, otherwise lstsq
            # Cholesky is faster and more stable for PD matrices
            L = np.linalg.cholesky(K)
            # Solve L v = y
            v = np.linalg.solve(L, y_std)
            # Solve L.T alpha = v
            alpha = np.linalg.solve(L.T, v)
        except np.linalg.LinAlgError:
            # Fallback to lstsq
            alpha, *_ = np.linalg.lstsq(K, y_std, rcond=None)
        
        def predict(X_new):
            X_new = np.asarray(X_new, dtype=np.float64)
            X_new_std = (X_new - mu) / sd
            K_new = matern52_kernel(X_new_std, X_std, l, sigma_f)
            y_pred_std = K_new @ alpha
            y_pred = y_pred_std * y_std + y_mean
            return y_pred
        
        return predict

    def make_predictor_poly(X_std, y_std, degree=5):
        """
        Fallback: Polynomial regression with regularization.
        """
        # Generate polynomial features
        # For d=1, degree=5 -> 6 features
        # For d=2, degree=5 -> (5+2)!/(5!2!) = 21 features
        # For d=3, degree=5 -> (5+3)!/(5!3!) = 56 features
        # This is manageable.
        
        def poly_features(X, degree):
            X = np.asarray(X)
            cols = [np.ones(len(X))]
            # Generate all monomials of degree <= 'degree'
            # Simple nested loops for small d
            if d == 1:
                for p in range(1, degree + 1):
                    cols.append(X[:, 0] ** p)
            elif d == 2:
                for p in range(0, degree + 1):
                    for q in range(0, degree - p + 1):
                        if p + q > 0:
                            cols.append((X[:, 0] ** p) * (X[:, 1] ** q))
            elif d == 3:
                for p in range(0, degree + 1):
                    for q in range(0, degree - p + 1):
                        for r in range(0, degree - p - q + 1):
                            if p + q + r > 0:
                                cols.append((X[:, 0] ** p) * (X[:, 1] ** q) * (X[:, 2] ** r))
            else:
                # Fallback for higher d (unlikely given task spec)
                # Just linear + squares
                for j in range(d):
                    cols.append(X[:, j])
                    cols.append(X[:, j] ** 2)
            return np.column_stack(cols)
        
        A = poly_features(X_std, degree)
        # Ridge Regression
        lam = 1e-2
        p = A.shape[1]
        ATA = A.T @ A
        ATy = A.T @ y_std
        try:
            w = np.linalg.solve(ATA + lam * np.eye(p), ATy)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(A, y_std, rcond=None)
            
        def predict(X_new):
            X_new = np.asarray(X_new, dtype=np.float64)
            X_new_std = (X_new - mu) / sd
            A_new = poly_features(X_new_std, degree)
            y_pred_std = A_new @ w
            y_pred = y_pred_std * y_std + y_mean
            return y_pred
            
        return predict

    best_predictor = None
    best_score = np.inf
    
    # Try GPR first
    if time.time() < DEADLINE - 50:
        try:
            # Initial guess for hyperparameters
            # l: length scale, sigma_f: signal variance, sigma_n: noise variance
            # Since y is normalized to std 1, sigma_f ~ 1, sigma_n ~ 0.01
            # l ~ 1.0 (in standardized space)
            
            def neg_log_likelihood(params):
                l = np.exp(params[0])
                sigma_f = np.exp(params[1])
                sigma_n = np.exp(params[2])
                
                # Clip to prevent extreme values
                l = np.clip(l, 1e-3, 10.0)
                sigma_f = np.clip(sigma_f, 1e-3, 10.0)
                sigma_n = np.clip(sigma_n, 1e-6, 1.0)
                
                dists = cdist(X_std, X_std, metric='euclidean')
                r = dists / l
                term1 = 1 + np.sqrt(5) * r + (5/3) * r**2
                term2 = np.exp(-np.sqrt(5) * r)
                K = sigma_f**2 * term1 * term2
                K += sigma_n**2 * np.eye(n)
                
                try:
                    # Compute log marginal likelihood:
                    # -0.5 * (y^T K^-1 y + log(det(K)) + n * log(2*pi))
                    L = np.linalg.cholesky(K)
                    # det(K) = det(L)^2 = prod(diag(L))^2
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    # Solve L v = y
                    v = np.linalg.solve(L, y_std)
                    # Solve L.T alpha = v -> alpha = K^-1 y
                    alpha = np.linalg.solve(L.T, v)
                    quad = y_std @ alpha
                    nll = 0.5 * (quad + log_det + n * np.log(2 * np.pi))
                    return nll
                except np.linalg.LinAlgError:
                    return 1e6
            
            # Initial params in log space
            x0 = np.array([np.log(1.0), np.log(1.0), np.log(0.01)])
            
            # Optimize
            result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', options={'maxiter': 200})
            
            if result.success and result.fun < 1e5:
                opt_params = result.x
                l = np.exp(opt_params[0])
                sigma_f = np.exp(opt_params[1])
                sigma_n = np.exp(opt_params[2])
                
                kernel_params = {'l': l, 'sigma_f': sigma_f, 'sigma_n': sigma_n}
                gpr_predictor = make_predictor_gpr(X_std, y_std, kernel_params)
                
                # Evaluate on training set to check quality (sanity check)
                y_pred_train = gpr_predictor(X)
                mse_train = np.mean((y_pred_train - y)**2)
                var_y = np.var(y)
                norm_mse = mse_train / var_y if var_y > 1e-12 else mse_train
                
                if norm_mse < 0.5: # Accept if reasonable
                    best_predictor = gpr_predictor
                    best_score = norm_mse
                
        except Exception as e:
            # GPR failed
            pass

    # If GPR failed or wasn't tried, try Polynomial
    if best_predictor is None or time.time() < DEADLINE - 10:
        poly_predictor = make_predictor_poly(X_std, y_std, degree=5)
        y_pred_train = poly_predictor(X)
        mse_train = np.mean((y_pred_train - y)**2)
        var_y = np.var(y)
        norm_mse = mse_train / var_y if var_y > 1e-12 else mse_train
        
        if best_predictor is None:
            best_predictor = poly_predictor
        else:
            # Compare with GPR
            if norm_mse < best_score:
                best_predictor = poly_predictor

    return best_predictor
