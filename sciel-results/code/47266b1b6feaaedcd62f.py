import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0
        
        # CMA-ES Parameters
        n = self.dim
        # Standard CMA-ES lambda
        lam = 4 + int(3 * np.log(n))
        
        # Initial parameters
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step sizes for evolution paths
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial function evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1
        
        # Main loop
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            ys = []
            for i in range(lam):
                if calls_made >= self.budget:
                    break
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset or use identity
                    C = np.eye(n)
                    L = np.eye(n)
                
                z = np.random.standard_normal(n)
                y = x + sigma * (L @ z)
                ys.append(y)
                calls_made += 1
            
            if not ys:
                break
                
            # Evaluate offspring
            fvals = [func(y) for y in ys]
            
            # Update best
            for y, f in zip(ys, fvals):
                if f < best_fx:
                    best_fx = f
                    best_x = y.copy()
            
            # Sort offspring by fitness
            indices = np.argsort(fvals)
            ys_sorted = [ys[i] for i in indices]
            
            # Update mean
            x_new = np.sum([weights[i] * ys_sorted[i] for i in range(mu)], axis=0)
            
            # Update sigma
            # Evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(L) * (L.T @ (x_new - x) / sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update covariance
            # Evolution path for C
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Rank-one and rank-mu updates
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (2 - c_c)) * C)
            
            # Rank-mu update
            yv = np.array([(y - x) / sigma for y in ys_sorted[:mu]])
            C += c_mu * np.sum([weights[i] * np.outer(yv[i], yv[i]) for i in range(mu)], axis=0)
            
            x = x_new
            
            # Optional: Restart if sigma is too small or too large
            if sigma < 1e-10 or sigma > 1e10:
                break

        return best_x, best_fx
