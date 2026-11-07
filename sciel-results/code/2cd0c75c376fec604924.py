"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """1+1-CMA-ES with adaptive covariance matrix."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        n = self.dim
        
        # Initialize
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # CMA-ES parameters for 1+1
        sigma = 2.0
        C = np.eye(n)  # Covariance matrix
        C_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Update rates
        mu = 1
        mu_w = 1
        c_sigma = (mu_w + 2) / (n + 4)
        d_sigma = 1 + 2 * max(0, (np.sqrt((mu_w + 1) / (mu_w + 3)) - 1) / (n + 2))
        c_c = (4 + mu_w / n) / (n + 4 + 2 * mu_w / n)
        c_1 = (2 - 1 / (n + 1.3) + mu_w / n) / ((n + 2) ** 2 + mu_w)
        c_mu = min(1 - c_1, 2 * (mu_w - 2 + 1 / mu_w) / ((n + 2) ** 2 + mu_w))
        
        # Path length normalization
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Sample from N(x, sigma^2 * C)
            z = rng.standard_normal(n)
            # Cholesky decomposition of C for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
            y = x + sigma * (L @ z)
            y = np.clip(y, -5, 5)
            fy = func(y)
            
            # Update paths and covariance
            y_norm = y - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * y_norm / sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (i + 1)) < (1.4 + 2 / (n + 1)) * chi_n)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_w) * y_norm / sigma
            
            if fy < fx:
                # Success: update mean and covariance
                delta = y - x
                x = y
                fx = fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = y.copy()
                
                # Update covariance
                C = (1 + c_1 * (2 - c_1 - c_mu) - c_1) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.outer(delta / sigma, delta / sigma)
                C_inv = np.linalg.inv(C)
                
                # Adapt sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            else:
                # Failure: only update paths
                pass
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-10):
                C = C + 1e-10 * np.eye(n)
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-4, 10.0)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
