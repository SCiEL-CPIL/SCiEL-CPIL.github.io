"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Compact CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 30s safety margin before 330s limit
        rng = np.random.default_rng(0)
        
        n = self.dim
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial state
        x = np.zeros(n)
        f0 = func(x)
        sigma = 2.0
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        best_x, best_f = x, f0
        calls = 1
        
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            Y = np.zeros((lambda_, n))
            for i in range(lambda_):
                # Cholesky factor of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes non-positive-definite
                    C = np.eye(n)
                    L = np.eye(n)
                    
                Y[i] = x + sigma * (B @ (D * (L @ rng.standard_normal(n))))
                Y[i] = np.clip(Y[i], -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls += lambda_
            
            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = Y[idx_best].copy()
            
            # Sort
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Mean update
            x_old = x
            x = sum(w[i] * Y[i] for i in range(mu))
            
            # Update CMA
            C_old = C
            C = ((1 - c_1 - c_mu) * C + 
                 c_1 * (2 - mu_c) * (np.outer(x - x_old, x - x_old) / sigma**2) + 
                 c_mu * sum(w[i] * np.outer(Y[i] - x_old, Y[i] - x_old) / sigma**2 for i in range(mu)))
            
            # Symmetrize to prevent drift
            C = (C + C.T) / 2
            
            # Path updates
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (D * ((x - x_old) / sigma)))
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x - x_old)
            
            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 1e3)
            
            # Eigenvalue decomposition
            try:
                D2, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D2, 1e-12))
            except np.linalg.LinAlgError:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
