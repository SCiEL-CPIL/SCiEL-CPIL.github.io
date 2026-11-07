import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(n))  # Population size
        mu = lambda_ // 2                 # Number of best individuals
        
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cs = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        
        # State
        m = np.zeros(n)
        C = np.eye(n)
        sigma = 2.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        for gen in range(1000):
            if calls >= budget:
                break
            if time.time() - start_time > 295:
                break
                
            # Precompute Cholesky factor for this generation
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes indefinite due to numerical issues
                C = (C + C.T) / 2 + 1e-9 * np.eye(n)
                L = np.linalg.cholesky(C)
                
            # Sample
            X = np.zeros((lambda_, n))
            F = np.zeros(lambda_)
            
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = np.random.randn(n)
                x = m + sigma * L @ z
                x = np.clip(x, -5, 5)
                X[i] = x
                F[i] = func(x)
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()
            
            if calls >= budget:
                break
                
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update Mean
            m_new = np.sum(w[:, None] * X[:mu], axis=0)
            
            # Update Paths
            # hsig condition
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - np.exp(-2 * (calls + 1) / cs)) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) / sigma * (L.T @ (m_new - m))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m_new - m) / sigma
            
            # Update Covariance
            rank1 = pc[:, None] * pc[None, :] + (hsig == 0) * cc * (2 - cc) * C
            rankmu = np.sum(w[:, None, None] * X[:mu, :, None] * X[:mu, None, :], axis=0)
            C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            
            # Symmetrize and regularize
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(n)
            
            m = m_new
            
            # Update Sigma
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)
            
        return best_x, best_f
