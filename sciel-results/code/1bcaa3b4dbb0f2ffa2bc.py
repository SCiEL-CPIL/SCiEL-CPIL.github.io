import numpy as np
import time
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s buffer for safety

    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    n, d = X.shape

    # Handle edge case: constant y
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        def predict_const(X_test):
            X_test = np.array(X_test, dtype=float)
            if X_test.ndim == 1:
                return np.array([y_mean])
            return np.full(X_test.shape[0], y_mean)
        return predict_const

    # Normalize X
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-10] = 1.0
    X_norm = (X - x_mean) / x_std

    # Normalize y
    y_norm = (y - y_mean) / y_std

    def _time_left():
        return deadline - time.time()

    # Matern 5/2 kernel with ARD
    def matern52_kernel(X1, X2, lengthscales):
        """Compute Matern 5/2 kernel matrix between X1 and X2."""
        X1_scaled = X1 / lengthscales
        X2_scaled = X2 / lengthscales
        sq_dists = cdist(X1_scaled, X2_scaled, 'sqeuclidean')
        sq_dists = np.maximum(sq_dists, 0.0)
        r = np.sqrt(sq_dists)
        sqrt3_r = np.sqrt(3.0) * r
        return (1.0 + sqrt3_r + 1.5 * sq_dists) * np.exp(-sqrt3_r)

    def loocv_error(lengthscales, sigma_f, sigma_n):
        """
        Compute LOOCV error for Kernel Ridge Regression with Matern 5/2 kernel.
        Returns the mean squared error.
        """
        K = matern52_kernel(X_norm, X_norm, lengthscales)
        K = sigma_f**2 * K
        A = K + sigma_n**2 * np.eye(n)
        
        try:
            # Solve A @ alpha = y_norm
            # Using Cholesky for stability
            L = np.linalg.cholesky(A + 1e-10 * np.eye(n))
            # alpha = A^{-1} y
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            
            # y_pred = K @ alpha
            y_pred = K @ alpha
            
            # Compute H = K @ A^{-1}
            # We need diag(H).
            # H = K @ A^{-1}
            # Let M = A^{-1}. We can solve A @ M = I.
            # Or, note that H = K @ A^{-1} = A^{-1/2} (A^{-1/2} K A^{-1/2}) A^{1/2}? No.
            # Simpler: Solve A @ X = K. Then X = A^{-1} K.
            # Since A is symmetric, A^{-1} is symmetric.
            # H = K A^{-1} = K X.T ? No.
            # X = A^{-1} K.
            # H = K A^{-1}.
            # If we compute X = solve(A, K), then X = A^{-1} K.
            # Then H = K X.T? No.
            # (K A^{-1})_{ij} = sum_k K_{ik} (A^{-1})_{kj}.
            # (X.T)_{ik} = X_{ki} = (A^{-1} K)_{ki} = sum_j (A^{-1})_{kj} K_{ji}.
            # This is not quite right.
            
            # Let's just compute the inverse M = A^{-1} using Cholesky.
            # M = L^{-T} L^{-1}
            # We already have L.
            # M = solve(L.T, solve(L, I))
            # This is O(N^3) again.
            
            # Alternative: 
            # H = K A^{-1}
            # diag(H)_i = sum_j K_{ij} (A^{-1})_{ji}
            # Let's solve A @ Y = K. Then Y = A^{-1} K.
            # Y_{ji} = (A^{-1} K)_{ji} = sum_k (A^{-1})_{jk} K_{ki}.
            # We want (A^{-1})_{ji}.
            # Note that A is symmetric, so A^{-1} is symmetric.
            # (A^{-1})_{ji} = (A^{-1})_{ij}.
            # From Y = A^{-1} K, we have Y_{ij} = sum_k (A^{-1})_{ik} K_{kj}.
            # This doesn't directly give (A^{-1})_{ij}.
            
            # Let's compute M = A^{-1} explicitly.
            # M = solve(L.T, solve(L, np.eye(n)))
            M = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
            
            # H = K @ M
            # diag(H) = np.einsum('ij,ji->i', K, M)
            h_diag = np.einsum('ij,ji->i', K, M)
            
            # LOOCV error
            # e_i = (y_i - y_pred_i) / (1 - h_i)
            denom = 1.0 - h_diag
            # Clamp denominator to avoid division by zero
            denom = np.maximum(np.abs(denom), 1e-8)
            # Preserve sign? For PD kernels, h_i < 1 usually.
            # If h_i > 1, it's unstable.
            
            errors = (y_norm - y_pred) / denom
            mse = np.mean(errors**2)
            
            return mse, alpha, K, L, M
        except:
            return np.inf, None, None, None, None

    best_mse = np.inf
    best_params = None
    best_alpha = None
    best_K = None
    best_L = None
    best_M = None
    best_ls = None
    best_sf = None
    best_sn = None

    # Grid Search
    # Length scales: logspace(-1, 1, 10) -> 0.1 to 10
    # Sigma_f: logspace(-1, 1, 5) -> 0.1 to 10 (since y is normalized, scale matters)
    # Sigma_n: logspace(-3, 0, 10) -> 0.001 to 1
    
    ls_candidates = np.logspace(-1, 1, 10)
    sf_candidates = np.logspace(-1, 1, 5)
    sn_candidates = np.logspace(-3, 0, 10)
    
    # For d>1, we use the same length scale for all dimensions initially (isotropic)
    # Then we can try ARD if time permits, but isotropic is often good enough and faster.
    # Given N=200, isotropic is a good start.
    
    for ls in ls_candidates:
        if _time_left() < 10:
            break
        lengthscales = np.full(d, ls)
        for sf in sf_candidates:
            if _time_left() < 5:
                break
            for sn in sn_candidates:
                if _time_left() < 2:
                    break
                mse, alpha, K, L, M = loocv_error(lengthscales, sf, sn)
                if mse < best_mse:
                    best_mse = mse
                    best_params = (ls, sf, sn)
                    best_alpha = alpha
                    best_K = K
                    best_L = L
                    best_M = M
                    best_ls = lengthscales
                    best_sf = sf
                    best_sn = sn

    # Refinement
    if best_params is not None and _time_left() > 30:
        ls0, sf0, sn0 = best_params
        ls_refine = np.logspace(np.log10(ls0) - 0.3, np.log10(ls0) + 0.3, 10)
        sf_refine = np.logspace(np.log10(sf0) - 0.3, np.log10(sf0) + 0.3, 10)
        sn_refine = np.logspace(np.log10(sn0) - 0.3, np.log10(sn0) + 0.3, 10)
        
        for ls in ls_refine:
            if _time_left() < 10:
                break
            lengthscales = np.full(d, ls)
            for sf in sf_refine:
                if _time_left() < 5:
                    break
                for sn in sn_refine:
                    if _time_left() < 2:
                        break
                    mse, alpha, K, L, M = loocv_error(lengthscales, sf, sn)
                    if mse < best_mse:
                        best_mse = mse
                        best_params = (ls, sf, sn)
                        best_alpha = alpha
                        best_K = K
                        best_L = L
                        best_M = M
                        best_ls = lengthscales
                        best_sf = sf
                        best_sn = sn

    # Fallback if no parameters found
    if best_alpha is None:
        # Use a simple default
        lengthscales = np.ones(d)
        sf = 1.0
        sn = 0.01
        K = matern52_kernel(X_norm, X_norm, lengthscales)
        K = sf**2 * K
        A = K + sn**2 * np.eye(n)
        try:
            best_alpha = np.linalg.solve(A, y_norm)
            best_ls = lengthscales
            best_sf = sf
            best_sn = sn
        except:
            best_alpha = np.zeros(n)
            best_ls = np.ones(d)
            best_sf = 1.0
            best_sn = 0.01

    def predict(X_test):
        X_test = np.array(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        X_test_norm = (X_test - x_mean) / x_std
        
        # Compute kernel between test and train
        K_test = matern52_kernel(X_test_norm, X_norm, best_ls)
        K_test = best_sf**2 * K_test
        
        y_pred_norm = K_test @ best_alpha
        y_pred = y_pred_norm * y_std + y_mean
        
        return y_pred

    return predict
