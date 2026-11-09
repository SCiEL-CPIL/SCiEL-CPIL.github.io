import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Lambda is population size
        # For n=5, lambda ~ 10 is standard
        lambda_ = 10
        mu = lambda_ // 2  # Number of selected individuals
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Mu_eff
        mu_eff = 1 / np.sum(weights**2)
        
        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initialization
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Time guard
        start_time = time.time()
        time_limit = 280  # 20s margin before 300s limit
        
        while calls < budget:
            if time.time() - start_time > time_limit:
                break
                
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular, reset or perturb
                C = np.eye(n)
                L = np.eye(n)
                
            # Generate lambda samples
            ys = np.random.randn(lambda_, n)
            pops = x_mean + sigma * (ys @ L.T)
            
            # Clip to box
            pops = np.clip(pops, -5, 5)
            
            # Evaluate
            fits = np.array([func(p) for p in pops])
            calls += lambda_
            
            # Update best
            min_idx = np.argmin(fits)
            if fits[min_idx] < best_f:
                best_f = fits[min_idx]
                best_x = pops[min_idx].copy()
                
            # Sort by fitness
            order = np.argsort(fits)
            ys_sorted = ys[order]
            pops_sorted = pops[order]
            fits_sorted = fits[order]
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * pops_sorted[:mu], axis=0)
            
            # Update evolution paths
            # p_sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (x_new - x_mean) / sigma)
            
            # h_sigma
            h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (budget - calls))) < (1 + 2 / (n + 1)) * np.sqrt(n + 1)
            
            # p_c
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(ys_sorted[:mu], ys_sorted[:mu]), axis=0)
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update mean
            x_mean = x_new
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            # eigvals = np.linalg.eigvalsh(C)
            # if np.min(eigvals) < 1e-10:
            #     C += 1e-10 * np.eye(n)
            
        return best_x, best_f
