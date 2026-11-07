import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = None
        best_fx = np.inf
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Initialize CMA-ES
        m = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        lam = 8
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff)))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + cs
        
        fx = evaluate(m)
        
        B, D = np.linalg.eigh(C)
        C_sqrt = B @ np.diag(D)
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            ys = []
            fs = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                ys.append(y)
                fs.append(fy)
                
            if len(ys) < lam:
                break
                
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            m_new = m + sigma * sum(weights[i] * (ys[i] - m) for i in range(mu))
            
            pc = (1 - cc) * pc + cc / np.sqrt(cc) * (m_new - m) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * damps) / np.sqrt(cc) * (m_new - m) / sigma
            
            delta_h = (1 - np.prod(1 - pc**2) ** cc) * (1 + (1 - np.prod(1 - ps**2) ** cs))
            C = (1 + c1 * delta_h - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            C += cmu * sum(weights[i] * np.outer(ys[i] - m, ys[i] - m) for i in range(mu))
            
            m = m_new
            
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - 1))
            
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))
            C_sqrt = B @ np.diag(D)

        return best_x, best_fx
