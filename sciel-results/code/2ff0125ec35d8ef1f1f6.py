"""Symbolic regression via multi-kernel ridge with per-dimension length scales and efficient CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def _rbf_kernel(X1, X2, length_scales):
    """Compute RBF kernel between two sets of points with per-dimension length scales."""
    X1s = X1 / length_scales
    X2s = X2 / length_scales
    sq1 = np.sum(X1s**2, axis=1)[:, None]
    sq2 = np.sum(X2s**2, axis=1)[None, :]
    sq_dists = sq1 + sq2 - 2.0 * (X1s @ X2s.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    return np.exp(-0.5 * sq_dists)


def _poly_kernel(X1, X2, degree, gamma, coef0):
    """Compute polynomial kernel."""
    return (gamma * (X1 @ X2.T) + coef0) ** degree


def _periodic_kernel(X1, X2, period, length_scale):
    """Compute periodic kernel."""
    diff = X1[:, None, :] - X2[None, :, :]
    sin_diff = np.sin(np.pi * diff / period)
    sq_sum = np.sum(sin_diff**2, axis=2)
    return np.exp(-2.0 * sq_sum / (length_scale**2))


def _compute_kernel(X1, X2, params):
    """
    Compute composite kernel: sum of RBF + polynomial + periodic kernels.
    params: dict with keys:
        - rbf_weight, rbf_ls (array of length d)
        - poly_weight, poly_degree, poly_gamma, poly_coef0
        - per_weight, per_period, per_ls
    """
    K = np.zeros((X1.shape[0], X2.shape[0]))
    
    if params.get('rbf_weight', 0) > 1e-12:
        K += params['rbf_weight'] * _rbf_kernel(X1, X2, params['rbf_ls'])
    
    if params.get('poly_weight', 0) > 1e-12:
        K += params['poly_weight'] * _poly_kernel(X1, X2, params['poly_degree'], 
                                                   params['poly_gamma'], params['poly_coef0'])
    
    if params.get('per_weight', 0) > 1e-12:
        K += params['per_weight'] * _periodic_kernel(X1, X2, params['per_period'], params['per_ls'])
    
    return K


def _kernel_ridge_solve(K, y, alpha_reg):
    """Solve kernel ridge regression: (K + alpha*I)^{-1} y"""
    n = K.shape[0]
    A = K + alpha_reg * np.eye(n)
    try:
        coeffs = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def _loocv_mse(K, y, alpha):
    """
    Compute LOOCV MSE efficiently using the hat matrix trick.
    LOOCV prediction for point i: y_i - K_ii / (1 + (K+alpha*I)^{-1}_{ii})
    MSE = mean((y_i - pred_i)^2)
    """
    n = K.shape[0]
    A = K + alpha * np.eye(n)
    try:
        # Solve for the inverse diagonal elements
        # We need diag((K + alpha*I)^{-1})
        # Use Cholesky for stability
        L = np.linalg.cholesky(A)
        # Solve L @ L.T @ X = I to get inverse, but we only need diagonal
        # More efficient: solve L @ X = I, then diag of X @ X.T
        # Actually, let's just compute the full inverse diagonal via solving
        # (K + alpha*I) @ coeffs = y for each basis vector is too slow
        # Use the formula: diag((K+alpha*I)^{-1}) can be obtained from
        # solving (K+alpha*I) @ Z = I, then taking diag(Z)
        # But that's O(n^3) anyway. Let's use a trick.
        
        # Alternative: compute all coefficients at once
        # coeffs = (K + alpha*I)^{-1} y
        # LOOCV pred_i = y_i - K_ii * coeffs_i / (1 + alpha * ... ) 
        # Actually the standard formula is:
        # pred_i^{(i)} = y_i - K_ii / h_ii where h_ii is the i-th diagonal of (K+alpha*I)^{-1} K
        # Or more simply: pred_i^{(i)} = y_i - K_ii * c_i where c = (K+alpha*I)^{-1} y
        # Wait, that's not quite right either.
        
        # The correct LOOCV formula for kernel ridge:
        # Let H = K(K + alpha*I)^{-1} be the hat matrix
        # pred_i^{(i)} = y_i - H_ii / (1 - H_ii) * (y_i - pred_i)
        # where pred_i is the in-sample prediction
        
        # Simpler approach: just compute it directly
        # coeffs_full = (K + alpha*I)^{-1} y
        # in_sample_pred = K @ coeffs_full
        # For LOOCV, the prediction without point i is:
        # y_i - K_ii * coeffs_i / (1 + alpha * something)
        
        # Let me use the direct formula:
        # (K + alpha*I) @ c = y
        # c = (K + alpha*I)^{-1} y
        # The leave-one-out prediction is:
        # y_i - K_ii * c_i / (1 + alpha * ((K+alpha*I)^{-1})_ii * K_ii / K_ii)
        
        # Actually, the standard efficient LOOCV for kernel ridge is:
        # pred_loo_i = y_i - K_ii / (1 + alpha * d_i) * c_i
        # where d_i is the i-th diagonal element of (K+alpha*I)^{-1}
        # No wait, let me derive it properly.
        
        # From the kernel ridge solution: c = (K + alpha*I)^{-1} y
        # The in-sample prediction is f(x_i) = K_i @ c = sum_j K_ij c_j
        # When we remove point i, the new solution c^{(i)} satisfies:
        # (K_{-i} + alpha*I) c^{(i)} = y_{-i}
        # By the Sherman-Morrison formula or block inversion, we can show:
        # f^{(i)}(x_i) = y_i - K_ii * c_i / (1 + alpha * ((K+alpha*I)^{-1})_ii)
        
        # Let me just compute it numerically for correctness
        coeffs = np.linalg.solve(A, y)
        in_sample_pred = K @ coeffs
        
        # Compute diagonal of (K + alpha*I)^{-1}
        # Solve A @ X = I, take diagonal
        # This is O(n^3) but we already did O(n^3) for the solve
        # We can get diag(A^{-1}) by solving A @ X = I
        # But that's expensive. Alternative: use the fact that
        # diag(A^{-1}) = diag(X) where A @ X = I
        
        # Actually, there's a simpler formula. Let me use:
        # h_ii = in_sample_pred_i - y_i + K_ii * c_i / (1 + alpha * diag(A^{-1})_i)
        
        # Let me just use the straightforward approach:
        # For each i, the LOOCV prediction is:
        # y_i - (K_ii / (1 + alpha * d_i)) * c_i
        # where d_i = ((K+alpha*I)^{-1})_ii
        
        # To get d_i efficiently, note that if A @ z = e_i, then z_i = d_i
        # But solving for each i is O(n^4) total.
        
        # Better: solve A @ Z = I once (O(n^3)), then diag(Z) gives all d_i
        # But forming I and solving is the same cost as the original solve.
        
        # Let's use a different approach: 
        # The LOOCV score can be computed as:
        # sum_i (y_i - f^{(i)}(x_i))^2
        # where f^{(i)}(x_i) = y_i - K_ii * c_i / (1 + alpha * w_i)
        # and w_i = ((K+alpha*I)^{-1})_ii
        
        # We can compute w_i by noting that:
        # If A @ c = y, then A @ (c / alpha) = y/alpha
        # And (K+alpha*I)^{-1} = (1/alpha)(I - K/alpha (K+alpha*I)^{-1})
        # This is getting circular.
        
        # Simplest correct approach: use the hat matrix
        # H = K @ (K + alpha*I)^{-1}
        # LOOCV pred_i = (y_i - H_ii * (y_i - in_sample_pred_i) / (1 - H_ii))
        # No, that's for linear regression.
        
        # For kernel ridge, the correct formula is:
        # f^{(i)}(x_i) = y_i - K_ii * c_i / (1 + alpha * ((K+alpha*I)^{-1})_ii)
        
        # Let me just compute ((K+alpha*I)^{-1})_ii by solving A @ X = I
        # Since we're already doing an O(n^3) solve, this adds another O(n^3)
        # But n=200, so n^3 = 8e6, which is fast.
        
        # Actually, we can be smarter. We already have the Cholesky factor L.
        # A = L @ L.T
        # A^{-1} = L^{-T} @ L^{-1}
        # diag(A^{-1}) = diag(L^{-T} @ L^{-1})
        # If we solve L @ Y = I, then Y = L^{-1}, and A^{-1} = Y.T @ Y
        # diag(A^{-1}) = sum_j Y[j,i]^2 = column norms of Y squared
        
        Y = np.linalg.solve(L, np.eye(n))
        diag_inv = np.sum(Y * Y, axis=0)
        
        # Now compute LOOCV predictions
        # f^{(i)}(x_i) = y_i - K_ii * c_i / (1 + alpha * diag_inv[i])
        K_diag = np.diag(K)
        loocv_pred = y - K_diag * coeffs / (1.0 + alpha * diag_inv)
        loocv_errors = y - loocv_pred
        return np.mean(loocv_errors ** 2)
        
    except np.linalg.LinAlgError:
        # Fallback: simple 5-fold CV
        n_folds = 5
        fold_size = n // n_folds
        indices = np.arange(n)
        cv_mses = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = min((fold + 1) * fold_size, n)
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            K_train_fold = K[train_idx][:, train_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            coeffs_fold = _kernel_ridge_solve(K_train_fold, y_train_fold, alpha)
            
            K_val = K[val_idx][:, train_idx]
            pred_val = K_val @ coeffs_fold
            mse = np.mean((pred_val - y_val_fold) ** 2)
            cv_mses.append(mse)
        return np.mean(cv_mses)


def _cv_score(X_norm, y_centered, params, alpha, n_folds=5):
    """Compute cross-validation MSE for given parameters using LOOCV."""
    K_full = _compute_kernel(X_norm, X_norm, params)
    return _loocv_mse(K_full, y_centered, alpha)


def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 310.0  # Leave 20s buffer for safety
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-12
    y_centered = y_train - y_mean
    
    # Normalize inputs to [0, 1]
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min + 1e-12
    X_norm = (X_train - x_min) / x_range
    
    best_params = None
    best_cv_mse = np.inf
    
    # Phase 1: Grid search over per-dimension RBF length scales and regularization
    # Start with isotropic, then try anisotropic for d > 1
    rbf_ls_grid = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    alpha_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    
    # First pass: isotropic length scales
    for ls_val in rbf_ls_grid:
        if time.time() > deadline:
            break
        for alpha in alpha_grid:
            if time.time() > deadline:
                break
            
            params = {
                'rbf_weight': 1.0, 'rbf_ls': np.full(d, ls_val),
                'poly_weight': 0.0, 'poly_degree': 2, 'poly_gamma': 1.0, 'poly_coef0': 1.0,
                'per_weight': 0.0, 'per_period': 2.0, 'per_ls': 1.0
            }
            
            cv_mse = _cv_score(X_norm, y_centered, params, alpha)
            if cv_mse < best_cv_mse:
                best_cv_mse = cv_mse
                best_params = params.copy()
                best_params['alpha'] = alpha
    
    if best_params is None:
        best_params = {
            'rbf_weight': 1.0, 'rbf_ls': np.ones(d),
            'poly_weight': 0.0, 'poly_degree': 2, 'poly_gamma': 1.0, 'poly_coef0': 1.0,
            'per_weight': 0.0, 'per_period': 2.0, 'per_ls': 1.0,
            'alpha': 1e-2
        }
    
    # Phase 2: Try anisotropic length scales if d > 1 and time permits
    if d > 1 and time.time() < deadline - 60:
        base_ls = best_params['rbf_ls'][0]
        # Try varying length scales per dimension
        for dim in range(d):
            if time.time() > deadline:
                break
            for ls_mult in [0.5, 0.7, 1.3, 2.0]:
                if time.time() > deadline:
                    break
                new_ls = best_params['rbf_ls'].copy()
                new_ls[dim] = base_ls * ls_mult
                params = best_params.copy()
                params['rbf_ls'] = new_ls
                
                cv_mse = _cv_score(X_norm, y_centered, params, params['alpha'])
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_params = params.copy()
    
    # Phase 3: Try adding polynomial kernel
    poly_configs = [
        {'weight': 0.1, 'degree': 2, 'gamma': 1.0, 'coef0': 1.0},
        {'weight': 0.5, 'degree': 2, 'gamma': 1.0, 'coef0': 1.0},
        {'weight': 1.0, 'degree': 2, 'gamma': 1.0, 'coef0': 1.0},
        {'weight': 0.5, 'degree': 3, 'gamma': 0.5, 'coef0': 1.0},
        {'weight': 0.5, 'degree': 4, 'gamma': 0.5, 'coef0': 1.0},
    ]
    
    for poly_cfg in poly_configs:
        if time.time() > deadline:
            break
        for ls_mult in [0.5, 1.0, 2.0]:
            if time.time() > deadline:
                break
            for alpha_mult in [0.1, 1.0, 10.0]:
                if time.time() > deadline:
                    break
                
                ls_val = best_params['rbf_ls'].copy() * ls_mult
                alpha = best_params['alpha'] * alpha_mult
                
                params = best_params.copy()
                params['rbf_ls'] = ls_val
                params['alpha'] = alpha
                params['poly_weight'] = poly_cfg['weight']
                params['poly_degree'] = poly_cfg['degree']
                params['poly_gamma'] = poly_cfg['gamma']
                params['poly_coef0'] = poly_cfg['coef0']
                
                cv_mse = _cv_score(X_norm, y_centered, params, alpha)
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_params = params.copy()
    
    # Phase 4: Try adding periodic kernel
    per_configs = [
        {'weight': 0.1, 'period': 2.0, 'ls': 0.5},
        {'weight': 0.5, 'period': 2.0, 'ls': 0.5},
        {'weight': 0.1, 'period': 1.0, 'ls': 0.3},
        {'weight': 0.5, 'period': 1.0, 'ls': 0.3},
        {'weight': 0.1, 'period': 5.0, 'ls': 1.0},
        {'weight': 0.5, 'period': 5.0, 'ls': 1.0},
    ]
    
    for per_cfg in per_configs:
        if time.time() > deadline:
            break
        
        params = best_params.copy()
        params['per_weight'] = per_cfg['weight']
        params['per_period'] = per_cfg['period']
        params['per_ls'] = per_cfg['ls']
        
        cv_mse = _cv_score(X_norm, y_centered, params, best_params['alpha'])
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_params = params.copy()
    
    # Phase 5: Fine-tune around best parameters
    if time.time() < deadline - 30:
        base_ls = best_params['rbf_ls'].copy()
        base_alpha = best_params['alpha']
        
        for ls_mult in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
            if time.time() > deadline:
                break
            for alpha_mult in [0.5, 0.7, 1.3, 2.0]:
                if time.time() > deadline:
                    break
                
                params = best_params.copy()
                params['rbf_ls'] = base_ls * ls_mult
                params['alpha'] = base_alpha * alpha_mult
                
                cv_mse = _cv_score(X_norm, y_centered, params, params['alpha'])
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_params = params.copy()
    
    # Final fit with best parameters
    final_params = best_params
    K_final = _compute_kernel(X_norm, X_norm, final_params)
    coeffs_final = _kernel_ridge_solve(K_final, y_centered, final_params['alpha'])
    
    def predict(X):
        X_test_norm = (X - x_min) / x_range
        K_test = _compute_kernel(X_test_norm, X_norm, final_params)
        return K_test @ coeffs_final + y_mean
    
    return predict
# EVOLVE-BLOCK-END
