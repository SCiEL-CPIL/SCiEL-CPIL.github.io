"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        # Step size
        sigma = 2.0
        
        # Mean
        x_mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Path lengths
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 + 2.0 / (n + 1.3)) / ((n + 2) ** 2 + mu_eff)
        c_mu = 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2) ** 2 + mu_eff)
        
        # Path vectors
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Cholesky factor
        C_sqrt = np.eye(n)
        
        # Expected norm of standard normal
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (256 * n ** 2))

        best_x = None
        best_f = np.inf
        
        calls_made = 0
        rng = np.random.default_rng(42)
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            current_lambda = min(lambda_, budget - calls_made)
            
            ys = []
            fs = []
            
            for i in range(current_lambda):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                ys.append(y)
                fs.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
            if len(fs) < 2:
                break

            order = np.argsort(fs)
            ys = np.array(ys)[order]
            fs = np.array(fs)[order]
            
            y_mean = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            v = (y_mean - x_mean) / sigma
            
            # Compute C_inv and C_inv_sqrt via eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ v)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv @ v)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            diffs = ys[:mu] - x_mean
            rank_mu_update = np.sum(weights[:, None, None] * (diffs[:, :, None] * diffs[:, None, :]), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu_update
            C = (C + C.T) / 2
            
            x_mean = y_mean
            
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_sqrt = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma = 1.0

        if best_x is None:
             best_x = np.zeros(n)
             best_f = func(best_x)
             
        return best_x, best_f
# EVOLVE-BLOCK-END
