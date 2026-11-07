import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        d = self.dim
        
        # CMA-ES Parameters
        lam = 9  # Population size
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        
        # Initial State
        m = np.zeros(d)
        sigma = 3.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Precompute constants
        h_sigma = (1 + 2 / (d + 1)) * (1 - c_sigma)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        best_x = m
        best_f = func(m)
        eval_count = 1
        rng = np.random.default_rng(0)
        
        while eval_count < self.budget:
            if time.time() - start_time > 290:
                break
            
            # Sample
            y = rng.multivariate_normal(np.zeros(d), C, size=lam)
            x = m + sigma * y
            
            # Evaluate
            f_vals = np.array([func(xi) for xi in x])
            eval_count += lam
            
            # Update Best
            idx_min = np.argmin(f_vals)
            if f_vals[idx_min] < best_f:
                best_f = f_vals[idx_min]
                best_x = x[idx_min]
                
            if eval_count >= self.budget:
                break
            
            # Sort
            order = np.argsort(f_vals)
            x_sorted = x[order]
            y_sorted = y[order]
            
            # New Mean
            m_new = np.dot(weights, x_sorted[:mu])
            
            # Step Size Update
            # y_mean = (m_new - m) / sigma
            y_mean = (m_new - m) / sigma if sigma > 1e-16 else np.zeros(d)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C @ y_mean
            norm_ps = np.linalg.norm(ps)
            h = 1 if norm_ps / (1 - (1 - c_sigma)**((eval_count + 1) * c_sigma)) < (1.4 + 2 / (d + 1)) * chi_n else 0
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Covariance Update
            pc = (1 - c_c) * pc + h * c_c * y_mean
            # Rank-1 update
            C = (1 + c_1 * (h - 1) * c_c * (2 - c_c) - c_1) * C + c_1 * np.outer(pc, pc)
            
            # Rank-Mu update
            # y_diff = y_sorted[:mu] - y_mean
            y_diff = y_sorted[:mu] - y_mean
            C += c_mu * np.dot(weights, (y_diff * np.sqrt(weights))[:, np.newaxis] * y_diff).T
            
            m = m_new
            
            # Ensure C is symmetric positive definite (numerical stability)
            C = (C + C.T) / 2
            # Regularize if needed? Usually not for small d.
            
        return best_x, best_f
