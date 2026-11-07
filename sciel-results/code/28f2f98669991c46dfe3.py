"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        
        # CMA-ES parameters
        weights = np.log(pop_size / 2) - np.arange(1, pop_size + 1)
        weights /= weights.sum()
        mu_eff = 1 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initial evaluation
        best_x = x.copy()
        best_f = func(best_x)
        calls_made = 1
        
        for gen in range(budget - 1):
            if time.time() > deadline:
                break
                
            # Generate population
            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular
                C = np.eye(dim)
                D = np.eye(dim)
                
            pop = np.zeros((pop_size, dim))
            for i in range(pop_size):
                pop[i] = x + sigma * (D @ rng.standard_normal(dim))
                pop[i] = np.clip(pop[i], -5, 5)
                
            # Evaluate
            fvals = np.zeros(pop_size)
            eval_done = 0
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                fvals[i] = func(pop[i])
                calls_made += 1
                eval_done += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()
            
            if eval_done < pop_size:
                # If we ran out of time mid-eval, we can't properly update CMA-ES
                # Just return best so far
                break

            # Sort
            order = np.argsort(fvals)
            x_mean = (pop[order[:int(mu_eff)]] * weights[:, None]).sum(axis=0)
            
            # Update step size control vectors
            D_inv = np.linalg.inv(D)
            pc = (1 - c_c) * pc + c_c * (D_inv @ (x_mean - x))
            norm_pc = np.linalg.norm(pc)
            hsig = (1 - np.exp(-1 + c_c)) * chi_n
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / norm_pc * (D_inv @ (x_mean - x))
            
            # Update mean
            x = x_mean
            
            # Update covariance matrix
            delta_hsig = (1 - np.exp(-1 + c_c)) * (1 - np.exp(-2 * c_c))
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) - C)
            
            # Add rank-mu update
            diff = (pop[order[:int(mu_eff)]] - x).T
            C += c_mu / mu_eff * (diff * (weights[:int(mu_eff)] ** 2)[None, :]) @ diff
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-10):
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
