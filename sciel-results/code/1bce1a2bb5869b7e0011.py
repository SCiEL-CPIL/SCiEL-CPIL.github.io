"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        pop_size = 15
        
        # Initialization
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        
        # Adaptation parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + mu_eff - 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # History for step size adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2) + 1 / (8 * dim**4))
        
        while calls_made + pop_size <= budget:
            # Generate population
            z = rng.standard_normal((pop_size, dim))
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
                
            pop = x_mean[None, :] + sigma * (z @ L.T)
            
            # Evaluate
            fitness = np.array([func(x) for x in pop])
            calls_made += pop_size
            
            # Sort
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]
            
            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = pop[0].copy()
                
            # Update Mean
            x_new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update Step Size
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new_mean - x_mean) / sigma
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximate h_sigma
            h_sigma = 1 if norm_p_sigma < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update Covariance
            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_new_mean - x_mean) / sigma
            delta_h = h_sigma * (1 - h_sigma) * c_c * (2 - c_c)
            
            delta_x = (pop[:mu] - x_mean[None, :]) / sigma
            outer_sum = np.sum(weights[:, None, None] * delta_x[:, :, None] * delta_x[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu + delta_h) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * outer_sum
            
            x_mean = x_new_mean
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
        # If there is a small remainder, do local search
        while calls_made < budget:
            step = 0.5 * sigma
            candidate = x_mean + step * rng.standard_normal(dim)
            cand_f = func(candidate)
            calls_made += 1
            if cand_f < best_f:
                best_f = cand_f
                best_x = candidate.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
