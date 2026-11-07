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
        # For dim=5, lambda=10 is a good choice.
        # 500 / 10 = 50 generations.
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        cs = (mueff + 1.5) / (dim + mueff + 3)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) - cs
        c1 = (2 / ((dim + 1.3)**2 + mueff)) * (1 - cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        
        # Initial state
        rng = np.random.default_rng(0)
        # Start at 0? Or random?
        # BBOB f6 has a shift. Random is safer.
        m = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        start_time = time.time()
        time_limit = 290.0
        
        best_x = m
        best_f = func(m)
        
        # Pre-compute inverse Cholesky of C (initially I)
        # We will update C and recompute Cholesky each generation.
        # For dim=5, this is very fast.
        
        for gen in range(budget // lam):
            if time.time() - start_time > time_limit:
                break
                
            # Sample population
            ys = np.zeros((lam, dim))
            xs = np.zeros((lam, dim))
            fs = np.zeros(lam)
            
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = L @ z
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                
                ys[i] = y
                xs[i] = x
                fs[i] = func(x)
                
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = x.copy()
                    
            # Sort
            order = np.argsort(fs)
            fs_sorted = fs[order]
            xs_sorted = xs[order]
            
            # Update mean
            m_new = np.sum(weights[:, np.newaxis] * xs_sorted[:mu], axis=0)
            
            # Update pc
            # y = (m_new - m) / sigma
            y_mean = (m_new - m) / sigma
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) * y_mean
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(pc) / chiN - 1))
            
            # Update C
            # Rank 1
            c1_term = c1 * (np.outer(pc, pc) - C)
            
            # Rank mu
            yw = (xs_sorted[:mu] - m) / sigma
            cmu_term = cmu * np.sum(weights[:, np.newaxis] * np.outer(yw, yw), axis=0)
            
            C = (1 - c1 - cmu) * C + c1_term + cmu_term
            C = (C + C.T) / 2
            
            m = m_new
            
        return best_x, best_f
