import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(d))
        if lam < 4: lam = 4
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff + 1) / (d + 1) - 1)) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Initial state
        mean = np.zeros(d)
        sigma = 2.0
        
        pc = np.zeros(d)
        ps = np.zeros(d)
        C = np.eye(d)
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        gen = 0
        
        while calls < budget and time.time() < deadline:
            # Generate population
            X = np.zeros((lam, d))
            F = np.zeros(lam)
            
            # Sample from N(mean, sigma^2 * C)
            # Using SVD for stability
            D, V = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                z = np.random.randn(d)
                X[i] = mean + sigma * (V * D) @ z
                X[i] = np.clip(X[i], -5, 5)
                F[i] = func(X[i])
                calls += 1
            
            if calls >= budget or time.time() > deadline:
                break
                
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Update mean
            mean_new = mean + sigma * np.sum(w[:mu, None] * X[:mu], axis=0)
            
            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / np.linalg.norm(np.sqrt(C)) * (mean_new - mean) / sigma
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((gen + 1) * c_sigma)) < (1.4 + 2 / (d + 1)) * np.trace(np.eye(d)) / d else 0
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(d) - 1))
            
            # Update C
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (mean_new - mean) / sigma
            
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            rank_mu = np.sum(w[:mu, None] * (X[:mu] - mean)[:, None] * (X[:mu] - mean), axis=0)
            
            C = (1 + c_1 * (c_c * (2 - c_c) - delta_hsig) - c_mu) * C \
                + c_1 * np.outer(pc, pc) \
                + c_mu * rank_mu
            
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(d)
            
            mean = mean_new
            gen += 1
            
        return best_x, best_f
