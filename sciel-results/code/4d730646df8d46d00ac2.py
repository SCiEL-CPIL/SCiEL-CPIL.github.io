"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 300.0
        
        n = self.dim
        pop_size = 4 + int(3 * np.log(n))
        
        x = np.zeros(n)
        sigma = 2.0
        
        # Initial population
        rng = np.random.default_rng(0)
        ys = np.array([x + sigma * rng.standard_normal(n) for _ in range(pop_size)])
        ys = np.clip(ys, -5, 5)
        fvals = np.array([func(y) for y in ys])
        
        best_x = ys[np.argmin(fvals)]
        best_f = np.min(fvals)
        
        # CMA-ES state
        C = np.eye(n)
        Cinv = np.eye(n)
        p_c = np.zeros(n)
        p_s = np.zeros(n)
        
        # Parameters
        c_sigma = (np.log(2) + 2 / (n + 3)) / (4 * n + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + c_sigma
        c_c = (4 + n / 4) / (n + 4 + 2 * n / (n + 4))
        c_1 = 2 / ((n + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu) / ((n + 2) ** 2 + mu))
        
        mu = pop_size // 2
        w = np.log(np.log(pop_size + 1) - np.log(np.arange(1, mu + 1)))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w ** 2)
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        calls = pop_size
        
        while calls < self.budget:
            # Time check
            if time.time() - start_time > timeout:
                break
                
            # Sort
            idx = np.argsort(fvals)
            y_w = ys[idx]
            w_vec = w
            
            # Mean update
            m = np.sum(w_vec[:, np.newaxis] * y_w, axis=0)
            
            # Covariance adaptation
            y_inv = np.dot(y_w - x, Cinv)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - c_c) * C - C) + c_mu * np.sum(w_vec[:, np.newaxis, np.newaxis] * np.outer(y_inv, y_inv), axis=0)
            
            # Invert
            try:
                Cinv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                Cinv = np.eye(n)
                
            # Path updates
            p_c = (1 - c_c) * p_c + c_c * np.dot(Cinv, m - x)
            norm_pc = np.linalg.norm(p_c)
            p_s = (1 - c_sigma) * p_s + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.dot(Cinv, m - x) / sigma
            norm_ps = np.linalg.norm(p_s)
            
            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Move mean
            x = m
            
            # Generate new population
            ys = np.array([x + sigma * rng.multivariate_normal(np.zeros(n), C) for _ in range(pop_size)])
            ys = np.clip(ys, -5, 5)
            fvals = np.array([func(y) for y in ys])
            calls += pop_size
            
            # Update best
            if np.min(fvals) < best_f:
                best_f = np.min(fvals)
                best_x = ys[np.argmin(fvals)]
                
        return best_x, best_f
# EVOLVE-BLOCK-END
