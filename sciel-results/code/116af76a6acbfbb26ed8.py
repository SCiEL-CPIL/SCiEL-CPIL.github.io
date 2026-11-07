import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        lam = 12
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)
        
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        
        rng = np.random.default_rng(0)
        m = rng.uniform(-2, 2, n)
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        calls = 0
        best_x = m
        best_f = func(best_x)
        calls += 1
        
        chi_lam = np.sqrt(n) * (1 - 1/(2*n) + 1/(4*n**2))
        
        gen = 0
        while calls + lam <= self.budget:
            if time.time() > deadline:
                break
                
            gen += 1
            
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-8 * np.eye(n)
                B = np.linalg.cholesky(C)
                
            Y = rng.standard_normal((lam, n))
            P = sigma * (Y @ B.T)
            X = m + P
            
            F = np.array([func(x) for x in X])
            calls += lam
            
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0]
                
            m_new = m + sigma * np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * (1 - cc)) * C_inv_sqrt @ (m_new - m) / sigma
            
            norm_ps = np.linalg.norm(ps)
            hc = 1 if norm_ps < (1.4 + 2/(n+1)) * (1 - cs) else 0
            
            # Corrected factor for pc
            # The term is sqrt(cc(2-cc)(1-cc^(2t)))
            # Note: 1-cc^(2t) is approx 1 for small t, but we compute it.
            # For large t, it converges to 1.
            sqrt_term = np.sqrt(cc * (2 - cc) * (1 - cc ** (2 * gen)))
            
            pc = (1 - cc) * pc + hc * sqrt_term * C_inv_sqrt @ (m_new - m) / sigma
            
            hs = 1 if norm_ps / chi_lam < (1.4 + 2/(n+1)) else 0
            
            update_ps = np.outer(ps, ps)
            Y_sel = (X_sorted[:mu] - m) / sigma
            update_mu = np.sum((w[:, np.newaxis] * Y_sel) * Y_sel, axis=0)
            
            factor = 1 + c1 * (hs - 1) + cmu
            C = factor * C - factor * (update_ps + update_mu)
            
            C = (C + C.T) / 2
            
            eigvals_new, _ = np.linalg.eigh(C)
            if np.min(eigvals_new) < 1e-12:
                C = C + 1e-6 * np.eye(n)
            
            sigma = sigma * np.exp(cs / (2 * chi_lam) * (norm_ps / chi_lam - 1))
            
            m = m_new
            
            if sigma < 1e-12:
                break

        return best_x, best_f
