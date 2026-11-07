"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """
    Black-box continuous optimizer using Gaussian Process Surrogate Modeling.
    
    Strategy:
    1. Initialize with a small set of random samples.
    2. Fit a Gaussian Process (GP) to the observed data.
    3. Select the next point by maximizing Expected Improvement (EI)
       over a pool of candidates.
    4. Iterate until budget or time limit is reached.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: return well before the 330s hard kill
        max_time = 310.0 
        
        rng = np.random.default_rng(42)
        
        # Hyperparameters for GP
        # Matern 5/2 kernel is often better for non-smooth functions
        # length_scale: characteristic scale of correlation
        # variance: signal variance
        # noise: observation noise
        
        # Storage for observations
        X_obs = []
        y_obs = []
        
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Initialization
        # Start with a reasonable number of points to get a rough landscape
        # 10 points is enough for a 5D problem; save budget for search
        n_init = 10
        for i in range(n_init):
            if time.time() - start_time > max_time:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            X_obs.append(x)
            y_obs.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        calls = len(X_obs)
        
        if best_x is None:
            # Fallback if even init failed (unlikely)
            x = np.zeros(dim)
            fx = func(x)
            return x, fx

        # Helper functions for GP computations
        def matern52_kernel(x1, x2, length_scale):
            """Compute Matern 5/2 kernel matrix."""
            # x1: (N, dim), x2: (M, dim)
            # Calculate squared Euclidean distances
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
            x1_norm = np.sum(x1**2, axis=1)[:, np.newaxis]
            x2_norm = np.sum(x2**2, axis=1)[np.newaxis, :]
            dist_sq = x1_norm + x2_norm - 2.0 * (x1 @ x2.T)
            dist_sq = np.maximum(dist_sq, 0) # numerical stability
            dist = np.sqrt(dist_sq)
            
            # Matern 5/2
            sqrt3_d = np.sqrt(3) * dist / length_scale
            K = (1.0 + sqrt3_d + (sqrt3_d**2) / 3.0) * np.exp(-sqrt3_d)
            return K

        def update_gp_hyperparams(X, y, ls, var, noise):
            """
            Simple coordinate descent/gradient step to adjust GP hyperparameters.
            We'll use a fixed set of sensible hyperparameters first, 
            but allow a light tuning if we have time. 
            For robustness and speed, we will stick to fixed hyperparameters 
            tuned for the box [-5, 5]^5.
            """
            return ls, var, noise

        # Fixed hyperparameters
        # Length scale: 1.0 seems reasonable for a box of width 10.
        # If the function varies rapidly, we might need smaller LS.
        # If it's smooth, larger LS. Matern 5/2 is flexible.
        length_scale = 1.5
        variance = 1.0
        noise = 1e-4
        
        # Normalize y for better numerical stability in GP
        # We will normalize y to mean 0, std 1 for the GP calculation,
        # then denormalize predictions.
        
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        y_mean = np.mean(y_obs)
        y_std = np.std(y_obs)
        if y_std < 1e-8:
            y_std = 1.0
        y_norm = (y_obs - y_mean) / y_std
        
        # Precompute K matrix and its inverse/Cholesky
        K = matern52_kernel(X_obs, X_obs, length_scale)
        K[np.diag_indices(len(K))] += noise
        
        # Cholesky decomposition for stable inversion
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # If singular, add more noise
            K[np.diag_indices(len(K))] += 1e-3
            L = np.linalg.cholesky(K)
            
        # Solve K * alpha = y_norm
        # K = L L^T => alpha = L^T \ (L \ y)
        Ly = np.linalg.solve(L, y_norm)
        alpha = np.linalg.solve(L.T, Ly)
        
        # Phase 2: Iterative Optimization with Acquisition Function
        # We will generate a pool of candidates and pick the best one.
        # Pool size: 150 candidates for better exploration
        # This balances exploration (pool) and exploitation (acquisition).
        
        n_pool = 150
        
        # Track when to switch to polish phase (last 20% of budget)
        polish_start = int(budget * 0.8)
        
        while calls < budget:
            if time.time() - start_time > max_time:
                break
                
            # Generate candidate pool
            # Mix of random points and perturbations around current best
            # to encourage exploitation if model is confident, 
            # but random points to ensure exploration.
            
            # 80% random, 20% local search around best
            n_random = int(n_pool * 0.8)
            n_local = n_pool - n_random
            
            candidates = np.zeros((n_pool, dim))
            
            # Random candidates
            candidates[:n_random] = rng.uniform(-5, 5, (n_random, dim))
            
            # Local candidates: perturb best_x
            # Sigma for local perturbation: 0.5
            local_sigma = 0.5
            if n_local > 0:
                candidates[n_random:] = best_x[None, :] + local_sigma * rng.standard_normal((n_local, dim))
                candidates[n_random:] = np.clip(candidates[n_random:], -5, 5)
            
            # Compute GP mean and variance for candidates
            # K_xs: (N_obs, N_cand)
            K_xs = matern52_kernel(X_obs, candidates, length_scale)
            
            # Mean: mu = K_xs^T @ alpha
            mu = K_xs.T @ alpha
            
            # Variance: 
            # var = variance - K_xs^T @ K^-1 @ K_xs
            # Using Cholesky: K^-1 = L^-T L^-1
            # K_xs^T @ L^-T is (N_cand, N_obs)
            v = np.linalg.solve(L, K_xs) # L v = K_xs
            # var = variance - sum(v^2, axis=1)
            var = variance - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-8) # Numerical stability
            std = np.sqrt(var)
            
            # Compute Expected Improvement (EI)
            # EI = (f_min - mu) * Phi(z) + std * phi(z)
            # where z = (f_min - mu) / std
            # Note: We are minimizing, so f_min is the current best normalized value.
            
            # Current best in normalized space
            # best_f is in original space.
            f_min_norm = (best_f - y_mean) / y_std
            
            # Handle std = 0
            z = (f_min_norm - mu) / std
            # Clip z to avoid numerical issues with extreme values
            z = np.clip(z, -10, 10)
            
            ei = (f_min_norm - mu) * norm.cdf(z) + std * norm.pdf(z)
            
            # Select candidate with highest EI
            best_idx = np.argmax(ei)
            x_new = candidates[best_idx].copy()
            
            # Evaluate new point
            fx_new = func(x_new)
            calls += 1
            
            # Update observations
            X_obs = np.vstack([X_obs, x_new])
            y_obs = np.append(y_obs, fx_new)
            
            # Update best
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
                
            # Refit GP
            # Update mean/std for normalization
            y_mean = np.mean(y_obs)
            y_std = np.std(y_obs)
            if y_std < 1e-8:
                y_std = 1.0
            y_norm = (y_obs - y_mean) / y_std
            
            K = matern52_kernel(X_obs, X_obs, length_scale)
            K[np.diag_indices(len(K))] += noise
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K[np.diag_indices(len(K))] += 1e-3
                L = np.linalg.cholesky(K)
                
            Ly = np.linalg.solve(L, y_norm)
            alpha = np.linalg.solve(L.T, Ly)
            
            # Optional: Adjust hyperparameters if we have time
            # For now, we stick to fixed hyperparameters for stability and speed.
            # We could do a simple grid search on length_scale every 50 calls,
            # but that adds complexity and risk.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
