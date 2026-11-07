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
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        # CMA-ES params
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_1 = (2 + 2 * mu_eff / (n + 1)) / (n + 1 + 3 * mu_eff / (n + 1))
        c_mu = (2 * (mu_eff - 2 + 1 / mu_eff)) / (n + 1 + mu_eff)
        
        sigma = 2.0
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        calls = 1
        
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            fs = []
            for _ in range(lam):
                z = np.random.standard_normal(n)
                # Cholesky decomposition for sampling
                try:
                    L = np.linalg.cholesky(C)
                    y = x + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    L = np.eye(n)
                    y = x + sigma * z
                    
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)
                
                if time.time() > deadline:
                    break
                    
            if len(ys) < lam:
                break
                
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = ys[idx_min].copy()
                
            # Sort
            idx_sorted = np.argsort(fs)
            ys_sorted = ys[idx_sorted]
            
            # New mean
            x_new = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            
            # Path updates
            y_mean = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.solve(C, y_mean)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Covariance update
            # Check for delta_sigma
            ps_norm = np.linalg.norm(ps)
            sigma_check = (1.4 + 2 / (n + 1)) * np.sqrt(n + 1 - mu_eff)
            delta_sigma = 1 if ps_norm / np.sqrt(1 - (1 - c_sigma)**(2 * calls / (n + 1))) < sigma_check else 0
            
            # Update C
            # C = (1 + c1*(delta_sigma - 1) - cmu)*C + c1*(pc*pc.T - C) + cmu*sum(w_i * (y_i - x_new)(y_i - x_new).T)
            # Note: Standard CMA-ES uses (y_i - x) or (y_i - x_new)?
            # Standard: (y_i - x) for the outer product term in some variants, but usually centered at x_new for stability?
            # Actually, the standard update is:
            # C <- (1 + c1*(delta_sigma - 1) - cmu)*C + c1*(pc*pc.T - C) + cmu * sum_i w_i * (y_i - x) (y_i - x).T
            # Wait, the term is usually (y_i - x_new) in some implementations to keep it centered?
            # Let's check the standard BBOB CMA-ES.
            # The standard formula uses (y_i - x) where x is the old mean.
            # However, using (y_i - x_new) is also common.
            # I will use the standard (y_i - x) formulation.
            
            delta_y = ys_sorted[:mu] - x
            rank_mu = np.sum(weights[:, None, None] * delta_y[:, :, None] * delta_y[:, None, :], axis=0)
            
            C = (1 + c_1 * (delta_sigma - 1) - c_mu) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definite
            try:
                # Check eigenvalues
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) <= 1e-6:
                    C = C + 1e-6 * np.eye(n)
            except:
                C = np.eye(n)
                
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / np.sqrt(n) - 1))
            
            x = x_new

        return best_x, best_f
