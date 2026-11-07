import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        rng = np.random.default_rng(12345)
        
        m = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        calls = 0
        best_x = m
        best_f = func(m)
        calls += 1
        
        # Main CMA-ES loop
        while calls + lam <= budget and time.time() < deadline:
            D_norm = rng.standard_normal((lam, n))
            BD = B * D
            X = m + sigma * (BD @ D_norm.T).T
            
            F = np.array([func(x) for x in X])
            calls += lam
            
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0]
                
            m_new = np.sum(w[:, None] * X_sorted[:mu], axis=0)
            y = (m_new - m) / sigma
            
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1.0 if norm_ps < (1.4 + 2.0/(n+1.0)) * np.sqrt((mu_eff+2.0)/mu_eff - 1.0) else 0.0
            
            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            
            h_c = 1.0 if (np.dot(ps, ps) / (1 - (1 - c_c)**2)) < (2 * (n + 1) + c_c) * (1 + c_c) else 0.0
            pc = (1 - c_c) * pc + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            C = (1 + c_1 * h_c * (mu_eff - 1) - c_1) * C + c_1 * np.outer(y, y)
            
            Y = (X_sorted[:mu] - m) / sigma
            C_mu = np.sum(w[:, None, None] * (Y[:, :, None] * Y[:, None, :]), axis=0)
            
            C = (1 + c_mu * h_c * (mu_eff - 1) - c_1 - c_mu) * C + c_mu * C_mu
            
            C = (C + C.T) / 2.0
            try:
                L = np.linalg.cholesky(C)
                D = np.diag(L)
                B = L / D[:, None]
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-12)
                B = vecs
                D = np.sqrt(vals)
                
            m = m_new
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(mu_eff) - 1))
            
        # Use remaining budget for (1+1)-ES or random search
        if calls < budget:
            # Simple (1+1)-ES refinement
            step = sigma * 0.5
            for _ in range(budget - calls):
                if time.time() >= deadline:
                    break
                y = np.clip(m + step * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y
                    m = y
                    step *= 1.2
                else:
                    step *= 0.85
                    
        return best_x, best_f
