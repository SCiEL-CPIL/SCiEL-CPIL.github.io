"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy
from scipy.stats import norm
import warnings

class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        def time_left():
            return deadline - time.time()
            
        # Phase 1: Initial Sampling (Latin Hypercube or Random)
        # We need at least 'dim + 2' points to have a non-degenerate covariance matrix for GP
        # Let's use 10-20 points for initialization to cover the space reasonably.
        init_points = 15
        if budget < init_points:
            init_points = budget
            
        # Simple random initialization within the box
        X_init = np.random.uniform(-5, 5, size=(init_points, dim))
        y_init = np.array([func(x) for x in X_init])
        calls_made += init_points
        
        # Update best
        min_idx = np.argmin(y_init)
        best_x = X_init[min_idx].copy()
        best_f = y_init[min_idx]
        
        # Data structures for GP
        X_data = X_init.copy()
        y_data = y_init.copy()
        
        # Phase 2: Bayesian Optimization Loop
        # We use a fixed number of candidate points to evaluate EI on, 
        # then pick the best candidate.
        # To keep it fast, we limit the number of candidate points.
        n_candidates = 200
        
        while calls_made < budget and time_left() > 1.0:
            try:
                # Fit Gaussian Process
                # We use a simple RBF kernel. 
                # Note: scipy's GP regression is not built-in in a simple class, 
                # so we implement a lightweight manual GP using matrix operations.
                
                # Standardize y to improve numerical stability? 
                # For BO, it's often better to work with raw y if scale is not huge, 
                # but standardizing helps kernel length scales.
                y_mean = np.mean(y_data)
                y_std = np.std(y_data) + 1e-8
                y_norm = (y_data - y_mean) / y_std
                
                # Kernel: RBF
                # We need to estimate length scales. For simplicity and speed, 
                # we can use a fixed length scale or do a very quick MLE.
                # Given the tight time budget and small N, let's use a fixed reasonable length scale 
                # or a simple heuristic. 
                # For dim=5 and box [-5,5], a length scale of 1.0-2.0 is often reasonable.
                # Let's try to adapt it slightly based on data spread if N is large enough, 
                # but for now, fixed L=1.5 is a safe bet for exploration.
                L = 1.5
                noise_var = 1e-4
                
                # Compute kernel matrix K
                # K[i,j] = exp(-0.5 * sum((xi-xj)^2) / L^2)
                sq_dists = np.sum(X_data**2, axis=1)[:, None] + np.sum(X_data**2, axis=1)[None, :] - 2 * X_data @ X_data.T
                sq_dists = np.maximum(sq_dists, 0)
                K = np.exp(-0.5 * sq_dists / (L * L))
                
                # Add noise for stability
                K[np.diag_indices_from(K)] += noise_var
                
                # Solve for alpha
                # alpha = K^-1 y_norm
                try:
                    alpha = np.linalg.solve(K, y_norm)
                except np.linalg.LinAlgError:
                    # If singular, add more noise
                    K[np.diag_indices_from(K)] += 1e-3
                    alpha = np.linalg.solve(K, y_norm)
                
                # Generate candidate points
                # We can sample candidates from a mixture: 
                # 1. Uniform random in the box
                # 2. Local perturbations around the current best
                # This helps with both exploration and exploitation.
                
                n_random = n_candidates // 2
                n_local = n_candidates - n_random
                
                X_rand = np.random.uniform(-5, 5, size=(n_random, dim))
                # Local perturbations: best_x + small noise
                # Scale noise to be small relative to the box, e.g., 0.5
                X_local = best_x + np.random.normal(0, 0.5, size=(n_local, dim))
                X_local = np.clip(X_local, -5, 5)
                
                X_cand = np.vstack([X_rand, X_local])
                
                # Compute GP mean and variance at candidates
                # k(x, X) = exp(-0.5 * sum((x - Xi)^2) / L^2)
                # We can compute this efficiently
                sq_dists_cand = np.sum(X_cand**2, axis=1)[:, None] + np.sum(X_data**2, axis=1)[None, :] - 2 * X_cand @ X_data.T
                sq_dists_cand = np.maximum(sq_dists_cand, 0)
                K_cand = np.exp(-0.5 * sq_dists_cand / (L * L))
                
                # Mean
                mu = K_cand @ alpha
                # Un-normalize mean
                mu = mu * y_std + y_mean
                
                # Variance
                # v = k(x, x) - k(x, X) K^-1 k(X, x)^T
                # k(x, x) = 1 for RBF
                v = 1.0 - np.sum(K_cand * (K_cand @ alpha), axis=1)
                # Un-normalize variance? 
                # var(y) = var(y_norm) * y_std^2
                v = np.maximum(v, 1e-8) * (y_std ** 2)
                std = np.sqrt(v)
                
                # Expected Improvement (EI)
                # EI = (f_best - mu) * Phi(z) + std * phi(z)
                # where z = (f_best - mu) / std
                # Note: We are minimizing, so f_best is the minimum.
                # Standard EI for minimization:
                # Improvement = f_best - f(x)
                # E[Improvement] = (f_best - mu) * Phi((f_best - mu)/std) + std * phi((f_best - mu)/std)
                
                z = (best_f - mu) / std
                EI = (best_f - mu) * norm.cdf(z) + std * norm.pdf(z)
                
                # Select best candidate
                best_cand_idx = np.argmax(EI)
                x_next = X_cand[best_cand_idx]
                
                # Evaluate
                f_next = func(x_next)
                calls_made += 1
                
                # Update data
                X_data = np.vstack([X_data, x_next.reshape(1, -1)])
                y_data = np.append(y_data, f_next)
                
                # Update best
                if f_next < best_f:
                    best_f = f_next
                    best_x = x_next.copy()
                    
            except Exception as e:
                # If GP fails (e.g., numerical issues), fall back to random search for a few steps
                # or break. To be safe, let's just do a random step.
                x_next = np.random.uniform(-5, 5, size=dim)
                f_next = func(x_next)
                calls_made += 1
                if f_next < best_f:
                    best_f = f_next
                    best_x = x_next.copy()
                
                # If we get too many errors, break to avoid wasting time
                if calls_made > budget - 5:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
