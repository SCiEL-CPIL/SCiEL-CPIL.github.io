"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time management: 330s limit, aim to finish by 280s to be safe
        start_time = time.time()
        deadline = start_time + 280.0

        # CMA-ES parameters
        # Population size lambda
        # Standard formula: lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + 3*1 = 7. Let's use 9 for better sampling.
        self.lambda_ = 4 + int(3 * np.log(dim))
        if self.lambda_ < 4:
            self.lambda_ = 4
        
        # Mu: number of parents
        self.mu = self.lambda_ // 2
        
        # Weights for recombination
        # Logarithmic weights
        weights = np.log(self.mu + 1) - np.log(np.arange(1, self.mu + 1))
        weights /= np.sum(weights)
        
        # Variance reduction factor mu_eff
        self.mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        cc = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        cmu = min(1 - c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff))
        
        # Initial state
        # Start at origin, which is often a good heuristic for BBOB instances if not specified
        # The problem states minimizer is in [-4, 4]^5.
        x_mean = np.zeros(dim)
        
        # Initial step size
        # Large enough to cover the space quickly, but not too large.
        # Domain is [-5, 5]^5.
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (for sampling)
        C_eigenvalues = np.ones(dim)
        C_eigenvectors = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        # Cumulation paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Counters
        n_evals = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Precompute sampling matrix
        def update_C_sqrt_inv():
            nonlocal C, C_eigenvalues, C_eigenvectors, C_sqrt_inv
            # Update eigen decomposition
            C_eigenvalues, C_eigenvectors = np.linalg.eigh(C)
            # Clip eigenvalues to prevent numerical issues
            C_eigenvalues = np.clip(C_eigenvalues, 1e-20, None)
            # Compute inverse square root
            C_sqrt_inv = C_eigenvectors @ np.diag(1.0 / np.sqrt(C_eigenvalues)) @ C_eigenvectors.T

        update_C_sqrt_inv()
        
        # Main loop
        while n_evals < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Sample population
            pop_x = np.zeros((self.lambda_, dim))
            for i in range(self.lambda_):
                # Standard normal
                z = np.random.standard_normal(dim)
                # Transform
                y = C_sqrt_inv @ z
                pop_x[i] = x_mean + sigma * y
                # Clip to domain? The problem says func clips, so we can pass raw values.
                # However, staying in bounds might help if the function is undefined outside?
                # "func(x) clips a finite length-5 vector to the box" -> So we don't need to clip ourselves.
            
            # Evaluate population
            pop_f = np.array([func(x) for x in pop_x])
            n_evals += self.lambda_
            
            # Update best
            idx_best = np.argmin(pop_f)
            if pop_f[idx_best] < best_f:
                best_f = pop_f[idx_best]
                best_x = pop_x[idx_best].copy()
            
            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x_sorted = pop_x[order]
            pop_f_sorted = pop_f[order]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * pop_x_sorted[:self.mu], axis=0)
            
            # Update step size (sigma)
            # Need to calculate sigma_old * sqrt(det(C_new)/det(C_old))
            # This is complex. A simpler approximation for small dim is to use the norm of the update.
            # Or use the standard CMA-ES update rule.
            
            # Calculate p_sigma
            # Need C^{-1/2} (x_mean_new - x_mean) / sigma
            p_sigma_update = C_sqrt_inv @ (x_mean_new - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * self.mu_eff) * p_sigma_update
            
            # Calculate h_sigma
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (n_evals + 1)) < (1 + 2 / (dim + 1)) * np.sqrt(1 + (1 - cs) ** (n_evals + 1))
            
            # Update p_c
            p_c = (1 - cc) * p_c
            if h_sigma:
                p_c += cc * C_sqrt_inv @ (x_mean_new - x_mean) / sigma
            
            # Update Covariance C
            # Rank-mu update
            v = (x_mean_new - x_mean) / sigma
            delta_h = (1 + (1 - h_sigma) * cc * (2 - cc)) * np.dot(p_c, C_sqrt_inv @ p_c) - 1
            rank_mu_update = np.sum([
                (weights[i] * np.outer(pop_x_sorted[i] - x_mean, pop_x_sorted[i] - x_mean) / sigma**2) 
                for i in range(self.mu)
            ])
            
            C = (1 + c1 * (delta_h - 1) + cmu) * C - c1 * np.outer(p_c, p_c) - cmu * rank_mu_update
            
            # Ensure positive definiteness (symmetrize and clip)
            C = (C + C.T) / 2
            # If C becomes non-PD, reset or repair
            try:
                eigvals, _ = np.linalg.eigh(C)
                if np.any(eigvals < 1e-20):
                    # Repair
                    C += 1e-20 * np.eye(dim)
            except:
                C = np.eye(dim)
            
            # Update eigen decomposition for next iteration's sampling
            update_C_sqrt_inv()
            
            # Update sigma
            # Standard CMA-ES: sigma = sigma * exp(cs/mu_eff * (norm(p_sigma)/chiN - 1))
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma = sigma * np.exp(cs / self.mu_eff * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clamp sigma
            if sigma > 1e10:
                sigma = 1e10
            if sigma < 1e-10:
                break # Converged
            
            x_mean = x_mean_new
            
            # Early stopping if sigma is very small or best_f is very good
            if best_f < 1e-5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
