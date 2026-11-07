import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        w = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        
        mueff = 1.0 / np.sum(w**2)
        
        cs = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi0 = np.sqrt(n) * (1 - 1.0 / (4 * n) - 1 / (25 * n * n))
        
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = m.copy()
        best_f = func(best_x)
        
        remaining = budget - 1
        gen = 0
        
        while remaining >= lam:
            gen += 1
            X = np.zeros((lam, n))
            F = np.zeros(lam)
            
            vals, vecs = np.linalg.eigh(C)
            vals = np.maximum(vals, 1e-12)
            sqrt_vals = np.sqrt(vals)
            inv_sqrt_vals = 1.0 / sqrt_vals
            
            for i in range(lam):
                z = np.random.randn(n)
                y = (vecs * sqrt_vals) @ z
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                X[i] = x
                F[i] = func(x)
                remaining -= 1
            
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                
            m_old = m.copy()
            m_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            pc = (1 - cc) * pc + cc * np.sqrt(mueff) * (m_new - m_old) / sigma
            
            norm_pc = np.linalg.norm(pc)
            thr = (1.4 + 2.0 / (n + 1)) * np.sqrt(1 + np.trace(C) / n)
            hs = 1.0 if norm_pc / np.sqrt(1 - (1 - cc) ** (2 * gen)) < thr else 0.0
            
            inv_sqrtC = vecs @ np.diag(inv_sqrt_vals) @ vecs.T
            ps = (1 - cs) * ps + hs * np.sqrt(cs * (2 - cs) * mueff) * inv_sqrtC @ (m_new - m_old) / sigma
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(cs / damps * (norm_ps / chi0 - 1))
            sigma = np.clip(sigma, 1e-9, 10.0)
            
            Y = (X_sorted[:mu] - m_old) / sigma
            Yw = w[:, np.newaxis] * Y
            RankMu = Yw.T @ Y
            
            delta_sigma = 2 * (1 - (1 - cc) ** 2) * (1 - (1 - cs) ** 2)
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs * hs * delta_sigma) * sigma**2 * C) + cmu * RankMu
            C = (C + C.T) / 2.0
            
            m = m_new
            
        if remaining > 0:
            x = best_x.copy()
            fx = best_f
            sig = 0.1
            for _ in range(remaining):
                y = np.clip(x + sig * np.random.randn(n), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sig *= 1.1
                else:
                    sig *= 0.9
            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
