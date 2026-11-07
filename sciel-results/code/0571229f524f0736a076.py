"""CMA-ES implementation for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # 10s safety margin before 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Initial setup
        x = np.zeros(d)
        f_x = func(x)
        
        # CMA-ES parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(d)))
        
        # Step size sigma
        sigma = 2.0
        
        # Covariance matrix C
        C = np.eye(d)
        
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Adaptation rates
        mu = lam // 2  # Number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Weights for the top mu individuals
        w = weights
        
        # Main loop
        calls = 1
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            ys = []
            f_ys = []
            for i in range(lam):
                z = np.random.randn(d)
                y = x + sigma * z @ C
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                f_ys.append(fy)
                calls += 1
                
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
            
            if not ys:
                break
            
            ys = np.array(ys)
            f_ys = np.array(f_ys)
            
            # Sort by fitness
            order = np.argsort(f_ys)
            ys = ys[order]
            f_ys = f_ys[order]
            
            # Update best solution
            if f_ys[0] < f_x:
                x = ys[0].copy()
                f_x = f_ys[0]
                
            # Select top mu
            if len(ys) < mu:
                mu_cur = len(ys)
                w_cur = w[:mu_cur]
                w_cur /= np.sum(w_cur)
            else:
                mu_cur = mu
                w_cur = w
                
            y_w = np.sum(w_cur[:, np.newaxis] * ys[:mu_cur], axis=0)
            
            # Update evolution paths
            z_w = np.linalg.solve(C, (y_w - x) / sigma)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1)))) / chi_n < (1.4 + 2 / (d + 1))
            
            if h_sigma:
                p_c = (1 - c_c) * p_c + c_c * z_w
            else:
                p_c = (1 - c_c) * p_c
            
            # Update mean
            x = y_w
            
            # Update covariance matrix
            v = p_c / np.sqrt(1 - (1 - c_c)**(2 * (calls + 1)))
            C = (1 - c_1 - c_mu) * C + c_1 * (v[:, np.newaxis] * v[np.newaxis, :] + np.eye(d))
            
            # Rank-mu update
            delta_h = (1 - h_sigma) * c_mu * mu_eff
            C = C + delta_h * np.sum(w_cur[:, np.newaxis, np.newaxis] * (
                (ys[:mu_cur] - x)[:, np.newaxis, :] @ (ys[:mu_cur] - x)[np.newaxis, :, :]
            ), axis=0)
            
            # Ensure positive semi-definite and symmetric
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)

        return x, f_x
# EVOLVE-BLOCK-END
