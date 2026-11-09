import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        mueff = 1 / np.sum(weights**2)
        
        cs = (mueff + 1.5) / (n + mueff + 3)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        c1 = (2 + 2/mueff) / ((n + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.linalg.cholesky(C)
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        best_fx = float('inf')
        best_x = m.copy()
        
        eval_count = 0
        
        while eval_count < budget:
            if time.time() > deadline:
                break
            
            ys = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = m + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            fvals = []
            for y in ys:
                if eval_count >= budget:
                    break
                fval = func(y)
                eval_count += 1
                fvals.append(fval)
                if fval < best_fx:
                    best_fx = fval
                    best_x = y.copy()
            
            if eval_count >= budget:
                break
            
            indices = np.argsort(fvals)
            ys_sorted = [ys[i] for i in indices]
            fvals_sorted = [fvals[i] for i in indices]
            
            m_old = m
            m = np.sum([weights[i] * ys_sorted[i] for i in range(mu)], axis=0)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (B.T @ (m - m_old) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (eval_count // lam + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs)**2)
            
            pc = (1 - cs) * pc + hsig * ds * cs * (m - m_old) / sigma
            
            v_sum = np.zeros((n, n))
            for i in range(mu):
                v = (ys_sorted[i] - m_old) / sigma
                v_sum += weights[i] * np.outer(v, v)
            
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * v_sum
            C = (C + C.T) / 2
            
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-6, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                B = np.linalg.cholesky(C)
            
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_fx
