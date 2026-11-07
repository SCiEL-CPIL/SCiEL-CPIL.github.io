import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        lam = 8
        mu = lam // 2
        
        weights = np.log(mu + 1) - np.log(1 + np.arange(mu))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        mean = np.zeros(self.dim)
        sigma = 2.0
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        c_sigma = (mueff + 2) / (self.dim + mueff + 5)
        c_c = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        c_mu = min(1 - c_sigma, 2 * (mueff - 2 + 1 / mueff) / (self.dim + 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + c_sigma
        
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        while calls_made < self.budget:
            if time.time() >= deadline:
                break
            
            # Ensure we don't exceed budget
            if calls_made + lam > self.budget:
                break
                
            # Generate offspring
            z = rng.standard_normal((lam, self.dim))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(C)
                vals = np.clip(vals, 1e-6, None)
                C = (vecs * vals) @ vecs.T
                L = np.linalg.cholesky(C)
                
            y = mean + sigma * (z @ L.T)
            y = np.clip(y, -5, 5)
            
            fy = np.array([func(yi) for yi in y])
            calls_made += lam
            
            idx_best = np.argmin(fy)
            if fy[idx_best] < best_f:
                best_f = fy[idx_best]
                best_x = y[idx_best].copy()
                
            idx_sorted = np.argsort(fy)
            y_sorted = y[idx_sorted]
            
            new_mean = np.dot(weights, y_sorted[:mu])
            
            diff = new_mean - mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * diff
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * diff
            
            threshold = (1.4 + 2 / (self.dim + 1)) * (1 - c_sigma)
            h_sigma = 1 if np.linalg.norm(p_sigma) / chi_n < threshold else 0
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            C = (1 - c_sigma + c_sigma * (1 - h_sigma)) * C + \
                c_sigma * h_sigma * np.outer(p_sigma, p_sigma)
            
            diff_matrix = (y_sorted[:mu] - mean) / sigma
            rank_mu_term = np.zeros((self.dim, self.dim))
            for i in range(mu):
                rank_mu_term += weights[i] * np.outer(diff_matrix[i], diff_matrix[i])
            C += c_mu * rank_mu_term
            
            C = (C + C.T) / 2
            mean = new_mean
            
        return best_x, best_f
