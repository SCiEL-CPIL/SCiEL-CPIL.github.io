import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        
        # Initial sigma
        sigma = 1.0
        
        # Initial mean
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Call counter
        calls = 0
        best_x = mean.copy()
        best_f = float('inf')
        
        # Helper to evaluate
        def safe_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            try:
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return float('inf')
        
        # Evaluate initial mean
        safe_eval(mean)
        
        # CMA-ES Constants
        c_sigma = (lam // 2 + 2) / (dim + lam // 2 + 5)
        d_sigma = 1 + 2 * max(0, (lam // 2 - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + lam // 2 / dim) / (dim + 4 + 2 * lam // 2 / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam // 2)
        c_mu = min(1 - c_1, 2 * ((lam // 2 - 2 + 1 / (lam // 2)) / (dim + 2) ** 2))
        
        # Weights for recombination
        mu = lam // 2
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu / 2 + 1) - np.log(i + 1)
        w = w / w.sum()
        
        # Effective population size
        mu_eff = 1 / np.sum(w ** 2)
        
        while calls < budget and time.time() < deadline:
            
            # 1. Sample
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim) + 1e-10 * np.eye(dim)
                L = np.linalg.cholesky(C)
                
            z = np.random.randn(lam, dim)
            points = mean + sigma * (L @ z.T).T
            
            # 2. Evaluate
            values = []
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                fx = safe_eval(points[i])
                values.append(fx)
                
            if len(values) < lam:
                break
                
            values = np.array(values)
            order = np.argsort(values)
            
            # 3. Update Mean
            old_mean = mean.copy()
            mean = np.sum(w[:, np.newaxis] * points[order[:mu]], axis=0)
            
            # 4. Update Paths
            # ps
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lam))) if calls > lam else 1.0
            # Simplified hs check: if norm(ps) < (1.4 + 2/(dim+1)) * sqrt((1-(1-cs)^2) * (1-(1-cs)^(2*gen))) * sigma
            # For simplicity, use a standard threshold
            if np.linalg.norm(ps) < (1.4 + 2.0 / (dim + 1.0)) * np.sqrt(1.0 - (1.0 - c_sigma) ** 2.0):
                hs = 1.0
            else:
                hs = 0.0
                
            ps = (1 - c_sigma) * ps + hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (mean - old_mean)
            
            # pc
            hc = 1.0 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2.0 / (dim + 1.0)) * np.sqrt(1.0 - (1.0 - c_c) ** 2.0) else 0.0
            pc = (1 - c_c) * pc + hc * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (mean - old_mean)
            
            # 5. Update Covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (hc * (2 - hc) - c_c) * C) + c_mu / sigma^2 * sum w_i * (x_i - old_mean) * (x_i - old_mean)^T
            delta_h = (1 - hc) * c_c * (2 - c_c)
            rank_one_term = np.zeros((dim, dim))
            for i in range(mu):
                diff = (points[order[i]] - old_mean) / sigma
                rank_one_term += w[i] * np.outer(diff, diff)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_h * C) + (c_mu / sigma ** 2) * rank_one_term
            
            # 6. Update Sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / sigma_norm_ps - 1))
            # sigma_norm_ps = E[||N(0, I)||] = sqrt(dim) * (1 - 1/(4dim) + 1/(21dim^2))
            sigma_norm_ps = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / sigma_norm_ps - 1))
            
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
