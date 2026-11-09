"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        rng = np.random.default_rng(0)

        # CMA-ES Parameters
        lambda_ = 8  # Population size
        mu = lambda_ // 2  # Number of best points
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initialization
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        L = np.eye(dim)  # Cholesky factor such that L @ L.T = C
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        budget_remaining = budget - 1
        t = 0
        
        # Pre-compute constants
        c_sigma_hat = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
        
        while budget_remaining >= lambda_:
            if time.time() > deadline:
                break
                
            # Sample
            z = rng.standard_normal((lambda_, dim))
            # Sample from N(0, C) using L
            y = z @ L.T
            x = x_mean + sigma * y
            
            # Clip
            x = np.clip(x, -5, 5)
            
            # Evaluate
            f_vals = np.array([func(xi) for xi in x])
            
            # Update best
            min_idx = np.argmin(f_vals)
            if f_vals[min_idx] < best_f:
                best_f = f_vals[min_idx]
                best_x = x[min_idx].copy()
                
            budget_remaining -= lambda_
            t += 1
            
            # Sort
            idx = np.argsort(f_vals)
            x_sorted = x[idx]
            f_sorted = f_vals[idx]
            
            # Update mean
            x_new = np.dot(w, x_sorted[:mu])
            
            # Update p_sigma
            y_mean = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_hat * y_mean
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma)**t)
            if denom < 1e-10:
                denom = 1e-10
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / denom - np.sqrt(dim)))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Update C
            # h_sigma condition
            h_sigma = 1.0 if (norm_p_sigma / denom < (1 + 2 / (dim + 1)) * c_sigma_hat) else 0.0
            c_1_eff = c_1 * (2 - h_sigma)
            
            # Rank 1 update
            C += c_1_eff * (np.outer(y_mean, y_mean) - (1 - h_sigma) * C)
            
            # Rank mu update
            y_mu = (x_sorted[:mu] - x_mean) / sigma
            C += c_mu * (np.outer(w, (y_mu * w).T @ y_mu) - C)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            
            # Update mean
            x_mean = x_new
            
            # Update Cholesky factor L
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-positive definite, reset to identity
                C = np.eye(dim)
                L = np.eye(dim)
                
        return best_x, best_f
# EVOLVE-BLOCK-END
