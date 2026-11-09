import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        rng = np.random.default_rng(0)
        
        # CMA-ES parameters
        n = self.dim
        lambda_ = 4 + int(3 * np.log(n)) # Population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # State
        x = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x
        best_f = func(x)
        calls = 1
        
        while calls + lambda_ <= self.budget:
            if time.time() - start_time > 280:
                break
                
            # Sample
            z = rng.standard_normal((lambda_, n))
            # Cholesky decomposition of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.eye(n)
            
            y = x + sigma * np.dot(z, B.T)
            
            # Evaluate
            f_y = np.array([func(yi) for yi in y])
            calls += lambda_
            
            # Sort
            idx = np.argsort(f_y)
            y = y[idx]
            f_y = f_y[idx]
            
            # Update best
            if f_y[0] < best_f:
                best_f = f_y[0]
                best_x = y[0].copy()
            
            # Weighted mean of selected
            y_w = np.dot(weights, y[:mu])
            
            # Update step size
            pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (y_w - x)
            if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 1/n) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(pc) / chi_n - 1))
            
            # Update covariance
            rank_mu = np.sum(weights[:, None] * (y[:mu] - x)[:, None] * (y[:mu] - x)[None, :], axis=0)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + c_sigma * (2 - c_sigma) * C) + c_mu * rank_mu
            
            # Update mean
            x = y_w
            
            # Ensure positive definiteness (optional, but good for stability)
            # C = 0.5 * (C + C.T)
            # Check eigenvalues
            # eigvals = np.linalg.eigvalsh(C)
            # if np.min(eigvals) < 1e-10:
            #     C += 1e-10 * np.eye(n)
            
        return best_x, best_f
