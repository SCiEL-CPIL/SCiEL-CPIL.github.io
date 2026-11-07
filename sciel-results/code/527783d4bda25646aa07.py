"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES Optimizer for BBOB f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial standard deviation
        sigma = 2.0  # Start with a relatively large step to explore [-5, 5]^5
        
        # Initial mean (center of the search)
        x_mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution path for covariance matrix adaptation
        p_c = np.zeros(dim)
        # Evolution path for step-size adaptation
        p_s = np.zeros(dim)
        
        # Decay rates
        mu = lambda_pop // 2  # Weights for the best mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_cov = (2 + 2 * mu_eff / dim) / (dim**1.8 + mu_eff + 13)
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        calls_used = 0
        remaining = budget
        
        # Store best solution found so far
        best_x = None
        best_f = np.inf
        
        # Main loop
        while remaining >= lambda_pop:
            # Generate offspring
            x_offspring = np.zeros((lambda_pop, dim))
            for i in range(lambda_pop):
                # Sample from multivariate normal: N(x_mean, sigma^2 * C)
                # Use Cholesky decomposition for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite (should be rare)
                    L = np.eye(dim)
                
                z = rng.standard_normal(dim)
                x_offspring[i] = x_mean + sigma * (L @ z)
                
                # Clip to bounds [-5, 5]
                x_offspring[i] = np.clip(x_offspring[i], -5, 5)
            
            # Evaluate offspring
            f_offspring = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if remaining <= 0:
                    break
                f_offspring[i] = func(x_offspring[i])
                calls_used += 1
                remaining -= 1
                
                if f_offspring[i] < best_f:
                    best_f = f_offspring[i]
                    best_x = x_offspring[i].copy()
            
            if remaining < 0:
                break
                
            # Sort by fitness
            order = np.argsort(f_offspring)
            x_sorted = x_offspring[order]
            f_sorted = f_offspring[order]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update step size
            # Compute inverse of sigma * C for step size adaptation
            # Approximation: use diagonal or simple norm
            # Standard CMA-ES uses:
            # p_s = (1 - c_sigma) * p_s + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma * ||x_mean - x_mean_old||) * (x_mean - x_mean_old) / sigma
            # Wait, the formula is:
            # p_s = (1 - c_sigma) * p_s + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_mean - x_mean_old) / (sigma * ||C^{-1/2} (x_mean - x_mean_old)||)
            # Let's use a simplified robust version:
            diff = x_mean - x_mean_old
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-16:
                # Approximate inverse square root of C
                # For simplicity and stability, we might just use the norm in the standard deviation
                # p_s = (1 - c_sigma) * p_s + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / (sigma * norm_diff)
                # This is a common simplification when C is close to identity or we want to avoid matrix inversion costs in every step
                p_s = (1 - c_sigma) * p_s + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / (sigma * norm_diff)
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_s) / chi_n - 1))
            else:
                p_s = (1 - c_sigma) * p_s
            
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_old) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
            
            # Rank-mu update
            C = (1 + c_cov * (c_c * (2 - c_c) + np.sum(weights[:mu]**2) - 1) - c_cov) * C
            
            # Add rank-one update from p_c
            C += c_cov * (1 + c_c * (2 - c_c) - 1) * np.outer(p_c, p_c)
            
            # Add rank-mu update
            for i in range(mu):
                y_i = (x_sorted[i] - x_mean_old) / sigma
                C += c_cov * weights[i] * np.outer(y_i, y_i)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Add small regularization if needed
            # C += 1e-10 * np.eye(dim)

            # If sigma becomes too small or too large, we might stop or adjust
            if sigma < 1e-10:
                break

        # If we have remaining budget, do local refinement around best_x
        if remaining > 0 and best_x is not None:
            # Simple local search with decreasing step size
            step_size = 0.1
            for _ in range(remaining):
                x = np.clip(best_x + step_size * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    step_size *= 0.9
                else:
                    step_size *= 0.95
                step_size = max(step_size, 0.001)

        return best_x, best_f
# EVOLVE-BLOCK-END
