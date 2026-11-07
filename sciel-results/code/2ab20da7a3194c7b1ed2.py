"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Gaussian Process Surrogate Optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        # Gaussian Process Parameters
        # Length scales: start with moderate values, allowing the GP to adapt
        # We use a simple Matern kernel with nu=2.5 or RBF. RBF is standard for smooth functions.
        # We will manually implement a simple GP update to avoid heavy dependencies if scipy.gp isn't ideal,
        # but scipy.stats is available. Let's use a simple RBF kernel implementation.
        
        # Hyperparameters: length_scale, variance
        # We'll use a simple adaptive scheme or fixed reasonable values for 5D box [-5,5]
        # Initial length scale ~ 2.0 (covers a good portion of the domain)
        length_scale = 2.0
        signal_variance = 100.0  # Scale of function values
        noise_variance = 1e-8    # Assume noiseless or very low noise
        
        # Data storage
        X_obs = []
        Y_obs = []
        
        # Initial sampling: Latin Hypercube or Stratified for good coverage
        n_init = 20
        # Simple stratified random sampling
        points_init = []
        for i in range(n_init):
            # Use random but ensure spread by sorting indices per dimension
            x = rng.uniform(-5, 5, d)
            points_init.append(x)
            
        # Actually, let's do a proper LHS for initial points
        # Divide domain into n_init intervals per dimension
        # For simplicity in code, we'll just use random but spread them out slightly
        # Or just random is fine for GP to start.
        
        # Evaluate initial points
        best_x = None
        best_f = np.inf
        calls = 0
        
        for x in points_init:
            if not time_ok(): break
            y = func(x)
            calls += 1
            X_obs.append(x)
            Y_obs.append(y)
            if y < best_f:
                best_f = y
                best_x = x.copy()
                
        if calls >= budget or not time_ok():
            return best_x, best_f

        X_obs = np.array(X_obs)
        Y_obs = np.array(Y_obs)
        
        # GP Kernel Function (RBF)
        def rbf_kernel(X1, X2, ls, var):
            # X1: (n1, d), X2: (n2, d)
            # Distance matrix
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * np.dot(X1, X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return var * np.exp(-0.5 * sq_dists / (ls**2))
        
        # Precompute K matrix
        K = rbf_kernel(X_obs, X_obs, length_scale, signal_variance)
        K += noise_variance * np.eye(len(X_obs))
        
        # Cholesky decomposition for stability
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter
            K += 1e-6 * np.eye(len(X_obs))
            L = np.linalg.cholesky(K)
            
        # Main Loop: Expected Improvement Acquisition
        # We will test a small set of candidate points and pick the one with max EI
        # Candidates: 
        # 1. Random points
        # 2. Points near current best (local exploitation)
        # 3. Points far from observed (exploration) - handled by GP uncertainty
        
        n_candidates = 50
        max_iter = budget - calls
        
        for i in range(max_iter):
            if not time_ok(): break
            if calls >= budget: break
            
            # Generate Candidates
            candidates = []
            
            # 1. Random exploration
            for _ in range(10):
                candidates.append(rng.uniform(-5, 5, d))
                
            # 2. Local exploitation around best point
            # Small step size
            for _ in range(20):
                step = rng.normal(0, 0.5, d)
                x_cand = best_x + step
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
                
            # 3. Intermediate points: mix of best and random
            for _ in range(20):
                alpha = rng.uniform(0.1, 0.5)
                x_rand = rng.uniform(-5, 5, d)
                x_cand = alpha * best_x + (1 - alpha) * x_rand
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
                
            candidates = np.array(candidates)
            
            # Compute GP Predictions for Candidates
            # K_star: (n_cand, n_obs)
            K_star = rbf_kernel(candidates, X_obs, length_scale, signal_variance)
            
            # Mean: mu = K_star @ K^-1 @ Y
            # Using Cholesky: K = L L^T, K^-1 Y = L^-T L^-1 Y
            alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
            mu = K_star @ alpha_vec
            
            # Variance: var = K(x,x) - K_star @ K^-1 @ K_star^T
            # K(x,x) = signal_variance for RBF
            var = signal_variance - np.sum(K_star * (K_star @ alpha_vec), axis=1)
            var = np.maximum(var, 1e-10) # Numerical stability
            
            std = np.sqrt(var)
            
            # Expected Improvement (EI)
            # EI = (f_best - mu - xi) * Phi(Z) + std * phi(Z)
            # where Z = (f_best - mu - xi) / std
            # We want to minimize, so f_best is the current minimum.
            # EI = (best_f - mu) * Phi((best_f - mu)/std) + std * pdf((best_f - mu)/std)
            # Note: Standard EI is for maximization. For minimization, we flip signs or use:
            # EI_min = (best_f - mu) * Phi((best_f - mu)/std) + std * pdf((best_f - mu)/std)
            # Wait, if mu < best_f, (best_f - mu) is positive.
            # Let's use the standard formula for minimization:
            # EI(x) = (f_best - mu(x)) * Phi( (f_best - mu(x)) / sigma(x) ) + sigma(x) * phi( (f_best - mu(x)) / sigma(x) )
            
            z = (best_f - mu) / std
            ei = (best_f - mu) * norm.cdf(z) + std * norm.pdf(z)
            
            # Select best candidate
            idx_best = np.argmax(ei)
            x_next = candidates[idx_best]
            
            # Evaluate
            y_next = func(x_next)
            calls += 1
            X_obs = np.vstack([X_obs, x_next])
            Y_obs = np.append(Y_obs, y_next)
            
            if y_next < best_f:
                best_f = y_next
                best_x = x_next.copy()
                
            # Update GP Model
            # Add jitter to avoid singularity
            K = rbf_kernel(X_obs, X_obs, length_scale, signal_variance)
            K += (noise_variance + 1e-6) * np.eye(len(X_obs))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-5 * np.eye(len(X_obs))
                L = np.linalg.cholesky(K)
                
        return best_x, best_f
# EVOLVE-BLOCK-END
