import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        start_time = time.time()
        time_limit = 280.0  # Safety margin before 300s kill
        
        def time_exceeded():
            return time.time() - start_time > time_limit
        
        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(d)))
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        # Initialize CMA-ES state
        x = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(20*d*d))
        
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1
        
        if calls >= self.budget or time_exceeded():
            return best_x, best_f
        
        # Run CMA-ES
        while calls < self.budget and not time_exceeded():
            # Generate population
            pops = []
            pop_f = []
            for i in range(lam):
                z = rng.standard_normal(d)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                pops.append(y)
                pop_f.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                if calls >= self.budget or time_exceeded():
                    break
            
            if len(pops) < 2:
                break
            
            # Sort by fitness
            sorted_idx = np.argsort(pop_f)
            pops = [pops[i] for i in sorted_idx]
            pop_f = [pop_f[i] for i in sorted_idx]
            
            # Update mean
            y_w = np.zeros(d)
            for i in range(mu):
                y_w += w[i] * (pops[i] - x)
            x_new = x + y_w
            
            # Update sigma
            y_sigma = np.zeros(d)
            for i in range(mu):
                y_sigma += w[i] * (B @ (D * (B.T @ (pops[i] - x))))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma / sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) - chi_n))
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c) ** 2) * np.eye(d)) \
                + c_mu * sum(w[i] * np.outer(pops[i] - x, pops[i] - x) for i in range(mu))
            
            # Recompute B and D
            eigenvalues, B = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            D = 1.0 / np.sqrt(eigenvalues)
            
            # Update mean
            x = x_new
            
            # Check for convergence
            if sigma < 1e-10:
                break
        
        return best_x, best_f
