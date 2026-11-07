import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 9
        mu = lambda_ // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        c_sigma = 1.0 / (mu_eff + dim + 2.0)
        c_c = 4.0 / (dim + 4.0)
        c_1 = 2.0 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 / ((dim + 2.0)**2 + mu_eff))
        d_sigma = 1.0 + 2.0 * max(0.0, (mu_eff - 1.0) / (dim + 2.0)) + c_sigma
        
        best_x = x.copy()
        best_fx = func(x)
        calls = 1
        
        while calls + lambda_ <= budget:
            if time.time() >= deadline:
                break
            
            # Generate
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            Y = np.array([x + sigma * (L @ np.random.randn(dim)) for _ in range(lambda_)])
            Y = np.clip(Y, -5, 5)
            
            F = np.array([func(y) for y in Y])
            calls += lambda_
            
            if F.min() < best_fx:
                best_fx = F.min()
                best_x = Y[np.argmin(F)].copy()
                
            sort_idx = np.argsort(F)
            Y = Y[sort_idx]
            
            x_old = x.copy()
            x = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update p_sigma
            C_inv_sqrt = np.linalg.inv(L)
            delta_sigma = (C_inv_sqrt @ (x - x_old)) / sigma
            p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * delta_sigma
            
            # Update p_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1.0 - (1.0 - c_sigma)**(2.0 * calls / lambda_))) < (1.4 + 2.0 / (dim + 1.0)) * np.sqrt(c_sigma * (2.0 - c_sigma))
            p_c = (1.0 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * delta_sigma
            c_c_eff = c_c * (2.0 - h_sigma * (1.0 - c_c))
            
            # Update C
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, np.newaxis] * np.outer(Y[:mu] - x_old, Y[:mu] - x_old) / (sigma**2), axis=0)
            C = (1.0 + c_1 * c_c_eff + c_mu - c_1 - c_mu) * C + c_1 * (rank_one - C) + c_mu * rank_mu
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1.0))
            sigma = max(1e-10, min(sigma, 10.0))
            
        # Use remaining budget for (1+1)-ES polish
        if calls < budget:
            sigma_local = max(sigma * 0.1, 0.1)
            x_local = best_x.copy()
            fx_local = best_fx
            window = 5
            succ = 0
            total = 0
            while calls < budget:
                if time.time() >= deadline: break
                y = np.clip(x_local + sigma_local * np.random.randn(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                if fy < fx_local:
                    succ += 1
                    x_local = y
                    fx_local = fy
                total += 1
                if total >= window:
                    rate = succ / total
                    if rate > 0.2: sigma_local *= 1.1
                    else: sigma_local *= 0.9
                    succ = 0
                    total = 0
                    sigma_local = max(1e-7, min(sigma_local, 1.0))
                    
        return best_x, best_fx
