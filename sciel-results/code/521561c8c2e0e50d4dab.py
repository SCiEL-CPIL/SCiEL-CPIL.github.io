import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim)) # Population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (2 + mu_eff / (dim + 2)) / (dim + 2 + 2 + mu_eff / dim)
        cmu1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu2 = min(1 - cmu1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        L = np.linalg.cholesky(C)
        
        # Time management
        start_time = time.time()
        deadline = start_time + 290
        
        calls = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls += 1
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Main Loop
        while calls < budget and time.time() < deadline:
            # 1. Sample
            X = np.zeros((lam, dim))
            F = np.zeros(lam)
            
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                    
                z = np.random.randn(dim)
                x_sample = x_mean + sigma * (L @ z)
                X[i] = x_sample
                
                # Evaluate with clipping
                x_eval = np.clip(x_sample, -5, 5)
                F[i] = func(x_eval)
                calls += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_eval.copy()
            
            # If we broke early due to budget/time, exit
            if calls >= budget or time.time() > deadline:
                break
                
            # 2. Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # 3. Update Mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(X[:mu] * weights[:, np.newaxis], axis=0)
            
            # 4. Update Path
            c_sigma = np.sqrt((1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) + cs)
            ps = (1 - cs) * ps + c_sigma * np.sqrt(mu_eff) / sigma * (x_mean - x_mean_old)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            
            pc = (1 - cc) * pc + hsig * cc * np.sqrt(mu_eff) * (x_mean - x_mean_old) / sigma
            
            # 5. Update Covariance
            # Using the simplified rank-mu update
            C = (1 - cmu1 - cmu2) * C + cmu1 * np.outer(pc, pc) + cmu2 * np.sum(weights[:, np.newaxis] * np.outer(X[:mu] - x_mean_old, X[:mu] - x_mean_old) / sigma**2, axis=0)
            
            # 6. Update Step Size
            sigma = sigma * np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls + 1))) - np.sqrt(dim * (1 + 1/dim))))
            
            # 7. Update Cholesky
            # Regularize C to ensure positive definiteness
            C = (C + C.T) / 2
            if np.linalg.cond(C) > 1e12:
                C += 1e-12 * np.eye(dim)
            
            try:
                L = np.linalg.cholesky(C)
            except:
                # Fallback if Cholesky fails
                L = np.eye(dim)
                C = np.eye(dim)
                sigma = max(sigma, 1e-4)
                pc = np.zeros(dim)
                ps = np.zeros(dim)

        return best_x, best_f
