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
        
        # CMA-ES parameters
        mu = max(1, dim // 2)
        lambda_ = 2 * mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        calls_made = 0
        best_x = x_mean.copy()
        best_fx = func(best_x)
        calls_made += 1
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Sample
            Y = np.zeros((lambda_, dim))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                
            for i in range(lambda_):
                z = np.random.randn(dim)
                Y[i] = x_mean + sigma * (L @ z)
                Y[i] = np.clip(Y[i], -5, 5)
                
            # Evaluate
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline or calls_made >= budget:
                    break
                F[i] = func(Y[i])
                calls_made += 1
                
            if calls_made >= budget:
                break
                
            # Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            if F_sorted[0] < best_fx:
                best_fx = F_sorted[0]
                best_x = Y_sorted[0].copy()
                
            # Update
            Y_parents = Y_sorted[:mu]
            x_mean_new = np.sum(weights[:, None] * Y_parents, axis=0)
            
            # Update ps
            diff = (x_mean_new - x_mean) / sigma
            try:
                u = np.linalg.solve(L.T, diff)
                v = np.linalg.solve(L, u)
                ps = (1 - c_sigma) * ps + c_sigma / np.sqrt(mu_eff) * v
            except np.linalg.LinAlgError:
                ps = np.zeros(dim)
                
            # Update pc
            pc = (1 - c_c) * pc + c_c / np.sqrt(mu_eff) * (x_mean_new - x_mean) / sigma
            
            # Update C
            y_diffs = (Y_parents - x_mean) / sigma
            rank_mu_term = np.sum(weights[:, None, None] * np.outer(y_diffs, y_diffs), axis=0)
            # Simplified rank-one term coefficient
            h_sigma = 1.0 
            c1_factor = 1 + h_sigma * (2 - c_c - c_c * (1 - c_sigma)) - c_c
            rank_one_term = c1_factor * np.outer(ps, ps)
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one_term + c_mu * rank_mu_term
            
            # Update sigma
            ps_norm = np.linalg.norm(ps)
            if ps_norm > (1 + c_sigma) * chi_n:
                sigma *= 1.2
            elif ps_norm < (1 - c_sigma) * chi_n:
                sigma *= 0.8
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            x_mean = x_mean_new
            C = (C + C.T) / 2
            
        return best_x, best_fx
