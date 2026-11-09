"""Symbolic regression via Sparse Gaussian Process (FITC) with adaptive inducing points and hyperparameter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """Fit a Sparse GP regression model and return predict callable."""
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin from 300s limit (5s buffer)
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better kernel behavior
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Precompute squared distances for train points
    # sq_dist[i, j] = ||X_norm[i] - X_norm[j]||^2
    X_sq = np.sum(X_norm**2, axis=1, keepdims=True)
    sq_dists = X_sq + X_sq.T - 2 * (X_norm @ X_norm.T)
    sq_dists = np.maximum(sq_dists, 0)
    
    def _fitc_predict(X_new, Z, K_ZZ, K_ZX, alpha, sigma2, l2, beta, noise):
        """
        FITC prediction.
        X_new: (m, d)
        Z: (M, d)
        K_ZZ: (M, M)
        K_ZX: (M, n)  # Kernel between Z and X_train
        alpha: (n,)   # Solution to (K_XZ K_ZZ^-1 K_ZX + noise I) alpha = y
        sigma2: float # Signal variance
        l2: float     # RBF length scale parameter (gamma)
        beta: float   # Output scale
        noise: float  # Noise variance
        """
        # Compute K_XZ_new: (m, M)
        X_new_norm = (X_new - X_min) / X_range
        # dists between X_new and Z
        X_new_sq = np.sum(X_new_norm**2, axis=1, keepdims=True)
        Z_sq = np.sum(Z**2, axis=1, keepdims=True)
        sq_dists_new = X_new_sq + Z_sq.T - 2 * (X_new_norm @ Z.T)
        sq_dists_new = np.maximum(sq_dists_new, 0)
        K_XZ_new = np.exp(-l2 * sq_dists_new)
        
        # Compute K_ZZ_new: (M, M)
        Z_sq_full = np.sum(Z**2, axis=1, keepdims=True)
        sq_dists_Z = Z_sq_full + Z_sq_full.T - 2 * (Z @ Z.T)
        sq_dists_Z = np.maximum(sq_dists_Z, 0)
        K_ZZ_new = np.exp(-l2 * sq_dists_Z)
        
        # Inverse of K_ZZ_new
        # Note: K_ZZ is already computed in fit, but l2 might change? 
        # In this implementation, we fix l2 during the solve step, so K_ZZ is constant for a given l2.
        # However, if we are re-using alpha from a specific l2, we must use the same l2 for prediction.
        # The alpha vector depends on l2. So we store l2 with the model.
        
        # Efficient computation:
        # f_star = K_XZ_new @ K_ZZ^{-1} @ (K_ZX @ alpha)
        # Let u = K_ZX @ alpha (M,)
        u = K_ZX @ alpha
        # v = K_ZZ^{-1} @ u
        # We can solve K_ZZ v = u
        v = np.linalg.solve(K_ZZ_new, u)
        f_star = K_XZ_new @ v
        
        return f_star * beta

    def _fitc_solve(Z, K_ZZ, K_ZX, y, noise):
        """
        Solve for alpha in FITC.
        alpha = (K_XZ K_ZZ^{-1} K_ZX + noise I)^{-1} y
        """
        n = len(y)
        M = Z.shape[0]
        
        # Efficiently compute K_eff = K_XZ K_ZZ^{-1} K_ZX
        # We can do this by solving K_ZZ W = K_ZX.T
        # Then K_eff = K_ZX.T @ W
        try:
            # Solve K_ZZ W = K_ZX.T
            # K_ZX is (M, n), so K_ZX.T is (n, M)
            W = np.linalg.solve(K_ZZ, K_ZX.T)
            K_eff = K_ZX.T @ W
            K_eff += noise * np.eye(n)
            alpha = np.linalg.solve(K_eff, y)
            return alpha
        except np.linalg.LinAlgError:
            return None

    best_score = np.inf
    best_params = None
    best_alpha = None
    best_K_ZZ = None
    best_K_ZX = None
    best_Z = None
    best_l2 = None
    best_noise = None
    best_beta = None
    
    # Hyperparameter grid
    # l2 (gamma) values for RBF kernel
    l2_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    # Noise values (since data is noiseless, small noise helps regularization)
    noise_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    # Number of inducing points
    M_values = [10, 20, 50]
    
    # Precompute Z candidates?
    # Z should be chosen from X_norm or sampled. 
    # For simplicity and speed, let's pick M random points from X_norm as Z.
    # Or use k-means? K-means is expensive. Random is fast.
    # Let's fix the random seed for reproducibility of Z selection.
    rng = np.random.RandomState(42)
    
    # Candidate configurations
    candidates = []
    for M in M_values:
        for l2 in l2_values:
            for noise in noise_values:
                candidates.append((M, l2, noise))
    
    # Sort candidates? Maybe larger M is better but slower.
    # Let's just iterate.
    
    n_candidates = len(candidates)
    
    # We need to evaluate each candidate.
    # For each candidate:
    # 1. Select Z (M points from X_norm)
    # 2. Compute K_ZZ (M, M)
    # 3. Compute K_ZX (M, n)
    # 4. Solve for alpha
    # 5. Compute LOO or CV score
    
    # To speed up selection of Z, we can pre-select a set of M_max points and use subsets?
    # Or just random each time. With M=50 and n=200, selection is cheap.
    
    # Optimization: Pre-compute K_ZZ for each (M, l2) pair?
    # Z depends on M. If we use the same random points for all M, we can reuse.
    # Let's select M_max = 50 random points.
    M_max = max(M_values)
    idx_Z_max = rng.choice(n, M_max, replace=False)
    Z_max = X_norm[idx_Z_max]
    
    # Pre-compute K_ZZ for each l2 for the max Z set?
    # K_ZZ_max[l2] = exp(-l2 * sq_dists_Z_max)
    Z_max_sq = np.sum(Z_max**2, axis=1, keepdims=True)
    sq_dists_Z_max = Z_max_sq + Z_max_sq.T - 2 * (Z_max @ Z_max.T)
    sq_dists_Z_max = np.maximum(sq_dists_Z_max, 0)
    
    K_ZZ_cache = {}
    for l2 in l2_values:
        K_ZZ_cache[l2] = np.exp(-l2 * sq_dists_Z_max)
    
    # Now iterate through candidates
    for i, (M, l2, noise) in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        
        # Get Z (first M points of Z_max)
        Z = Z_max[:M]
        
        # Get K_ZZ (top-left MxM of K_ZZ_cache[l2])
        K_ZZ = K_ZZ_cache[l2][:M, :M]
        
        # Compute K_ZX (M, n)
        # dists between Z and X_norm
        Z_sq = np.sum(Z**2, axis=1, keepdims=True)
        sq_dists_ZX = Z_sq + X_sq.T - 2 * (Z @ X_norm.T)
        sq_dists_ZX = np.maximum(sq_dists_ZX, 0)
        K_ZX = np.exp(-l2 * sq_dists_ZX)
        
        # Solve for alpha
        alpha = _fitc_solve(Z, K_ZZ, K_ZX, y, noise)
        if alpha is None:
            continue
        
        # Compute validation score
        # Use 2-fold CV for speed, or LOO approximation?
        # FITC LOO is complex. Let's do a simple split.
        # Split indices
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        
        half = n // 2
        train_idx = idx[:half]
        val_idx = idx[half:]
        
        # Refit on train_idx
        y_train_fold = y[train_idx]
        X_train_fold = X_norm[train_idx]
        
        # K_ZX_fold: (M, half)
        # We already have K_ZX (M, n). Extract columns.
        K_ZX_fold = K_ZX[:, train_idx]
        
        alpha_fold = _fitc_solve(Z, K_ZZ, K_ZX_fold, y_train_fold, noise)
        if alpha_fold is None:
            continue
        
        # Predict on val_idx
        # K_ZX_val: (M, half)
        K_ZX_val = K_ZX[:, val_idx]
        
        # Prediction: f_val = K_XZ_val @ K_ZZ^{-1} @ (K_ZX_fold @ alpha_fold)
        u = K_ZX_fold @ alpha_fold
        v = np.linalg.solve(K_ZZ, u)
        # K_XZ_val is (half, M)
        # K_XZ_val = K_ZX_val.T
        f_val = K_ZX_val.T @ v
        
        mse_val = np.mean((f_val - y[val_idx])**2)
        
        if mse_val < best_score:
            best_score = mse_val
            # Store best parameters
            # We need to re-solve on full dataset for final prediction
            # But we can just store the alpha from the full solve we did earlier?
            # No, the alpha from the full solve used all y.
            # The alpha from the fold solve is different.
            # We should re-solve on full dataset to get the final alpha.
            # Or we can just use the alpha we already computed for the full set?
            # Yes, we computed `alpha` for the full set above.
            # So we can use that `alpha`.
            
            best_params = {'M': M, 'l2': l2, 'noise': noise}
            best_alpha = alpha
            best_K_ZZ = K_ZZ
            best_K_ZX = K_ZX
            best_Z = Z
            best_l2 = l2
            best_noise = noise
            best_beta = 1.0 # We assume y is scaled by beta. Since we fit alpha on y, beta is effectively 1.
            # Wait, if y has variance, should we scale?
            # The kernel outputs are in the scale of the features.
            # If we don't include an output scale parameter, the model might not fit the amplitude.
            # Let's add an output scale parameter `beta`.
            # y = beta * f + noise
            # alpha = (K_eff + noise I)^-1 y
            # If we include beta, we need to optimize it too.
            # For now, let's assume beta=1 and see if it works.
            # If not, we can add beta to the search.
            
    # If no model found, fallback to simple RBF kernel ridge regression
    if best_params is None:
        # Fallback
        l2 = 1.0
        noise = 1e-3
        K_train = np.exp(-l2 * sq_dists)
        A = K_train + noise * np.eye(n)
        best_alpha = np.linalg.solve(A, y)
        best_params = {'fallback': True}
        
        def predict(X_new):
            X_new = np.asarray(X_new, dtype=np.float64)
            X_new_norm = (X_new - X_min) / X_range
            X_new_sq = np.sum(X_new_norm**2, axis=1, keepdims=True)
            sq_dists_new = X_new_sq + X_sq.T - 2 * (X_new_norm @ X_norm.T)
            sq_dists_new = np.maximum(sq_dists_new, 0)
            K_test = np.exp(-l2 * sq_dists_new)
            return K_test @ best_alpha
        
        return predict
    
    # Define predict function for FITC
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        X_new_norm = (X_new - X_min) / X_range
        
        # Compute K_XZ_new: (m, M)
        m = X_new_norm.shape[0]
        Z = best_Z
        M = Z.shape[0]
        
        Z_sq = np.sum(Z**2, axis=1, keepdims=True)
        X_new_sq = np.sum(X_new_norm**2, axis=1, keepdims=True)
        sq_dists_new = X_new_sq + Z_sq.T - 2 * (X_new_norm @ Z.T)
        sq_dists_new = np.maximum(sq_dists_new, 0)
        K_XZ_new = np.exp(-best_l2 * sq_dists_new)
        
        # u = K_ZX @ alpha
        u = best_K_ZX @ best_alpha
        
        # v = K_ZZ^{-1} @ u
        # We can pre-compute the inverse of K_ZZ or the Cholesky factor
        # Since K_ZZ is small (M <= 50), solving is fast.
        # But we can pre-compute K_ZZ_inv for speed in prediction.
        # Let's do it here once.
        # Actually, we can just solve each time, it's very fast for M=50.
        v = np.linalg.solve(best_K_ZZ, u)
        
        f_star = K_XZ_new @ v
        return f_star

    return predict
# EVOLVE-BLOCK-END
