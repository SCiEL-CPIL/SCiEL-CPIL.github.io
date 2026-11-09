import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        # Population size for CMA-ES
        N = 4 + int(3 * np.log(dim))
        if N > self.budget // 2:
            N = 4
            
        mu = N // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff + 1) / (dim + 1) - 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / (dim**1.8 + 10) + c_sigma
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim**1.8 + 18))
        
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1
        budget_left = self.budget - 1
        gen = 0
        
        while budget_left >= N and time.time() < deadline:
            gen += 1
            z = np.random.standard_normal((N, dim))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                
            X = x_mean + sigma * (z @ L.T)
            
            F = np.array([func(x) for x in X])
            calls_made += N
            budget_left -= N
            
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                
            idx = np.argsort(F)
            X_sorted = X[idx]
            
            x_new = np.sum(w[:, None] * X_sorted[:mu], axis=0)
            step = (x_new - x_mean) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ step)
            norm_p_sigma = np.linalg.norm(p_sigma)
            
            # Simplified h_sigma
            h_sigma = 1 if norm_p_sigma < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff + 1 - 1) else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (L.T @ step)
            norm_p_c = np.linalg.norm(p_c)
            h_c = 1 if norm_p_c < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff + 1 - 1) else 0
            
            C = (1 - c_1 - c_mu + c_c + c_c * c_sigma - h_c * c_c * c_sigma) * C \
                + c_1 * (np.outer(p_c, p_c) - C) \
                + c_mu * np.sum(w[:, None, None] * np.outer(X_sorted[:mu] - x_mean, X_sorted[:mu] - x_mean) / sigma**2, axis=0)
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
            
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T
            
            x_mean = x_new
            
        return best_x, best_f
