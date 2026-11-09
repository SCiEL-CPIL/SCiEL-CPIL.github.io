"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        n = self.dim
        # Standard CMA-ES parameters
        lambda_pop = 4 + int(3 * np.log(n))
        # Weights
        w = np.log(lambda_pop / 2 + 1) - np.log(np.arange(1, lambda_pop + 1))
        w /= np.sum(w)
        mu = int(np.sum(w > 0))  # Usually all are positive, but standard practice
        w_mu = w[:mu]
        mu_eff = (np.sum(w_mu) ** 2) / np.sum(w_mu ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial state
        # Start at center of the box [0,0,0,0,0]
        m = np.zeros(n)
        sigma = 2.0  # Initial step size, covers roughly [-4, 4]
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Cholesky factor of C
        C_chol = np.eye(n)
        
        best_x = m.copy()
        best_f = func(m)
        evals = 1
        
        # Pre-compute constants
        d_norm = 1 + 2 / (n + 1)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Sample
            z = np.random.randn(lambda_pop, n)
            x = np.zeros((lambda_pop, n))
            for i in range(lambda_pop):
                x[i] = m + sigma * (z[i] @ C_chol.T) # z * C_chol.T is equivalent to C_chol * z.T for symmetric C
                # Note: np.linalg.cholesky returns L such that L L^T = C.
                # So we want x = m + sigma * L * z.
                # If z is row vector, x = m + sigma * (z @ L.T)
            
            # Evaluate
            f_vals = np.array([func(xi) for xi in x])
            evals += lambda_pop
            
            # Update best
            idx_min = np.argmin(f_vals)
            if f_vals[idx_min] < best_f:
                best_f = f_vals[idx_min]
                best_x = x[idx_min].copy()
                
            # Sort
            order = np.argsort(f_vals)
            x_sorted = x[order]
            f_sorted = f_vals[order]
            
            # Update mean
            m_new = m + sigma * np.sum(w_mu[:, None] * x_sorted[:mu], axis=0)
            
            # Update evolution paths
            y_sigma = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (self.budget - evals))) < (1.4 + 2 / (n + 1)) * chi_n)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Update step size (sigma)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_1 = np.outer(p_c, p_c) + delta_h * C
            rank_mu = np.sum(w_mu[:, None, None] * np.outer(x_sorted[:mu] - m, x_sorted[:mu] - m) / sigma**2, axis=0)
            
            C = (1 + c_1 + c_mu - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            # Wait, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            m = m_new
            
            # Update Cholesky factor
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-positive definite, add small identity
                C += 1e-10 * np.eye(n)
                C_chol = np.linalg.cholesky(C)
                
            # Termination check
            if sigma < 1e-12:
                break
            # If the population spread is very small
            if np.max(f_sorted) - np.min(f_sorted) < 1e-12 * max(1, abs(np.min(f_sorted))):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
