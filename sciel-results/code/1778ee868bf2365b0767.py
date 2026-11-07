import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # State
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x
        best_f = func(x)
        calls = 1
        
        start_time = time.time()
        deadline = start_time + 250
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sample
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
            except:
                L = np.eye(n)
                
            Y = np.random.randn(lambda_, n) @ L.T
            X = x + sigma * Y
            
            Fs = np.array([func(xi) for xi in X])
            calls += lambda_
            
            idx_min = np.argmin(Fs)
            if Fs[idx_min] < best_f:
                best_f = Fs[idx_min]
                best_x = X[idx_min].copy()
                
            order = np.argsort(Fs)
            Y_sorted = Y[order]
            
            # Update mean
            x_new = x + sigma * (weights @ Y_sorted[:mu])
            
            # Update paths
            # C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            
            # Update Cov
            delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.linalg.norm(p_c) ** 2 / n
            rank_one = np.outer(x_new - x, x_new - x) / sigma ** 2
            rank_mu = sum(weights[i] * np.outer(Y_sorted[i], Y_sorted[i]) for i in range(mu))
            
            C = (1 + c_1 * (delta_h - 1) + c_mu) * C - c_1 * np.outer(p_c, p_c) - c_mu * rank_mu
            C = (C + C.T) / 2 + 1e-12 * np.eye(n)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            x = x_new
            
        return best_x, best_f
