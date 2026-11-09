"""Combined optimizer: global scan + CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global scan followed by CMA-ES with adaptive covariance."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Phase 1: Global scan to find promising basins
        n_scan = 50
        xs = rng.uniform(-5, 5, size=(n_scan, self.dim))
        fs = np.full(n_scan, np.inf)
        
        calls_made = 0
        for i in range(n_scan):
            if time.time() > deadline:
                break
            fx = func(xs[i])
            fs[i] = fx
            calls_made += 1
        
        if calls_made == 0:
            # Fallback if even scan fails (shouldn't happen)
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            return x, fx

        # Sort scan results
        sorted_idx = np.argsort(fs)
        
        best_x = xs[sorted_idx[0]].copy()
        best_f = fs[sorted_idx[0]]

        # Phase 2: CMA-ES on the best point found
        # CMA-ES maintains a population and adapts the covariance matrix
        # to model the correlation structure of the landscape.
        
        # Initialize CMA-ES parameters
        x_mean = best_x.copy()
        sigma = 1.0
        dim = self.dim
        
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        
        # CMA-ES parameters
        alpha_mu = 0.5 * (1 - 1.0 / (2 * lam))
        c_sigma = (1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1)) - 1)) / (dim + 3 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1)) - 1))
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        mu = lam // 2
        
        # Weights for recombination
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Covariance matrix
        C = np.eye(dim)
        # Cholesky decomposition for sampling
        C_inv_sqrt = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial step size
        sigma = 1.0
        
        # Maximum iterations for CMA-ES
        max_iter = 50
        
        for it in range(max_iter):
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Sample population
            pop = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Apply Cholesky decomposition
                pop[i] = x_mean + sigma * (C_inv_sqrt @ z)
                # Clip to bounds
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate population
            f_vals = np.full(lam, np.inf)
            for i in range(lam):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break
                f_vals[i] = func(pop[i])
                calls_made += 1
            
            if calls_made == 0:
                break
                
            # Sort by fitness
            sorted_f = np.argsort(f_vals)
            
            # Update best
            if f_vals[sorted_f[0]] < best_f:
                best_f = f_vals[sorted_f[0]]
                best_x = pop[sorted_f[0]].copy()
            
            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(w[:, np.newaxis] * pop[sorted_f[:mu]], axis=0)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_mean - x_old) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma
            
            # Update covariance matrix
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * np.trace(C_inv_sqrt) else 0
            
            C = (1 + c_c * (2 - c_c) * h_sigma - c_c) * C + c_c * np.outer(p_c, p_c)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / (dim + 1) * (np.linalg.norm(p_sigma) / np.trace(C_inv_sqrt) - 1))
            
            # Update Cholesky decomposition
            # Simple approach: recompute Cholesky every iteration
            try:
                C_inv_sqrt = np.linalg.cholesky(C)
            except:
                # If C is not positive definite, reset
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                sigma = 1.0
                break
            
            # Stop if sigma is too small
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
