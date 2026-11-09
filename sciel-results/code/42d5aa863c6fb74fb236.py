import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        
        # CMA-ES Parameters
        lambda_pop = 10
        mu = lambda_pop // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Initial values
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Update parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (np.sqrt((mu_eff + 1) / (n + 1) - 1) - 1) / (n + 3))
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        calls = 0
        best_x = x.copy()
        best_f = func(best_x)
        calls += 1
        
        # Pre-compute expected norm of standard normal
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(32*n**2))
        
        while calls < budget:
            # Time guard
            if time.time() - start_time > 290:
                break
                
            # Sample population
            ys = []
            fs = []
            
            # Compute Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                
            # Determine how many points we can evaluate
            max_points = min(lambda_pop, budget - calls)
            
            for _ in range(max_points):
                z = np.random.randn(n)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)
            
            if not ys:
                break
                
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Sort
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Recombination
            # Use only top mu points if we have at least mu, else use all
            m_use = min(mu, len(ys))
            w_use = w[:m_use]
            # Renormalize weights if m_use < mu
            w_use = w_use / w_use.sum()
            
            y_w = np.sum(w_use[:, None] * ys[:m_use], axis=0)
            
            # Update mean
            x_old = x.copy()
            x = y_w
            
            # Update sigma path
            if sigma > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x - x_old) / sigma
            else:
                p_sigma = np.zeros(n)
                
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Update covariance path
            # Heuristic h_c
            h_c = 1.0 if norm_p_sigma / chi_n < 1.4 + c_c else 0.0 # Using p_sigma norm as proxy for p_c norm is common simplification, or compute p_c first?
            # Standard CMA-ES computes p_c, then uses its norm for h_c.
            # Let's compute p_c first.
            
            if sigma > 1e-16:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma
            else:
                p_c = np.zeros(n)
                
            norm_p_c = np.linalg.norm(p_c)
            # Recalculate h_c based on p_c
            h_c = 1.0 if norm_p_c / chi_n < 1.4 + c_c else 0.0
            
            # Update covariance matrix
            # Heuristic h_sigma for C update
            h_sigma = 1.0 if norm_p_sigma / chi_n < 1.4 + c_sigma else 0.0
            
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_sigma)**2) * (1 - h_sigma) * np.eye(n))
            
            for i in range(m_use):
                diff = ys[i] - x_old
                C += c_mu * w_use[i] * np.outer(diff, diff)
                
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(n)
            
        return best_x, best_f
