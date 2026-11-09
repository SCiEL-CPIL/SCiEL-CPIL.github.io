"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with time-deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        rng = np.random.default_rng(12345)
        
        # CMA-ES parameters
        n = self.dim
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff / (n + 1.7) + c_c) / ((n + 1.3) ** 2 + mueff + c_c)
        c_mu = (2 - c_1 + mueff / (n + 1.7)) / ((n + 1.3 + 2 * mueff / n) ** 2 + mueff - c_1 + c_c)
        c_mu = min(c_mu, 2 * (1 - c_1) / (n - 1))
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))

        # Initialize
        x_mean = rng.uniform(-5, 5, n)
        sigma = 2.0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        calls_made = 1
        max_iter = self.budget - 1
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Generate population
            ys = np.empty((lambda_, n))
            for i in range(lambda_):
                z = rng.standard_normal(n)
                ys[i] = x_mean + sigma * (B @ (D * z))
                
            # Evaluate population
            fs = np.empty(lambda_)
            for i in range(lambda_):
                x_i = np.clip(ys[i], -5, 5)
                fs[i] = func(x_i)
                calls_made += 1
                
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = x_i.copy()
                    
                if time.time() > deadline:
                    break
            
            if calls_made >= self.budget:
                break
                
            # Sort
            idx = np.argsort(fs)[:mu]
            y_weighted = np.sum(weights[:, np.newaxis] * ys[idx], axis=0)
            y_mean = np.sum(weights * fs[idx])
            
            # Update mean
            x_mean = x_mean + c_1 * (pc - x_mean) + c_mu * (y_weighted - x_mean)
            
            # Update covariance
            y_diff = ys[idx] - x_mean
            delta_s = (np.linalg.norm(y_weighted) / (sigma * chi_n) > (1.4 + 2 / (n + 1)) * c_sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * y_weighted
            pc = (1 - c_c) * pc + c_c * (y_weighted / sigma)
            
            C = np.eye(n)
            C += c_1 * (np.outer(pc, pc) - np.eye(n)) + c_mu * np.sum(weights[:, np.newaxis] * np.outer(y_diff, y_diff), axis=0)
            if delta_s:
                C += (1 - c_1) * c_sigma * d_sigma * np.outer(ps, ps)
                
            # Eigendecomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clamp sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-8, 100)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
