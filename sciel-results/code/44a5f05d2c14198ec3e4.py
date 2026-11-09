"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        # Try multiple restarts with different initializations
        max_restarts = 15
        budget_per_restart = max(20, self.budget // max_restarts)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with different strategies
            if restart == 0:
                x = np.zeros(self.dim)
            elif restart == 1:
                x = rng.uniform(-4, 4, self.dim)
            elif restart == 2:
                x = rng.uniform(-2, 2, self.dim)
            else:
                # Use previous best as starting point with perturbation
                if best_x is not None:
                    x = np.clip(best_x + rng.normal(0, 1.0, self.dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)
            
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Run CMA-ES for this restart
            sigma = 1.0
            lambda_ = 4 + int(3 * np.log(self.dim))
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = (2 / ((self.dim + 1.3) ** 2 + mu_eff)) * (1 - c_c)
            c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1 / mu_eff)) / 
                       ((self.dim + 2) ** 2 + mu_eff))
            chi = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
            
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            C = np.eye(self.dim)
            step_size = np.ones(self.dim)
            
            budget_used = 0
            stagnation = 0
            
            while budget_used < budget_per_restart:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                ys = []
                fs = []
                for i in range(lambda_):
                    if time.time() > deadline:
                        break
                    y = x + sigma * (np.dot(C, rng.standard_normal(self.dim)))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    ys.append(y)
                    fs.append(fy)
                    budget_used += 1
                
                if len(ys) < 2:
                    break
                
                # Sort by fitness
                idx = np.argsort(fs)
                ys = np.array(ys[idx])
                fs = np.array(fs[idx])
                
                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = ys[0].copy()
                
                # Selection
                y_mu = np.sum(weights[:, None] * ys[:mu], axis=0)
                
                # Update mean
                x_old = x.copy()
                x = y_mu
                
                # Update step size (CSA)
                y_sigma = np.dot(C, (y_mu - x_old) / sigma)
                p_sigma = (1 - c_c) * p_sigma + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_c) ** (budget_used + 1))) / chi < 1.4 + 2 / (self.dim + 1)
                
                # Update evolution path
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_mu - x_old) / sigma
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.sum(weights[:, None] * np.outer(ys[:mu] - x_old, ys[:mu] - x_old), axis=0) / sigma ** 2
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Ensure positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
                
                # Update step size
                sigma *= np.exp((c_1 / (5 * self.dim)) * (np.linalg.norm(p_sigma) / chi - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Check for stagnation
                if fs[0] >= fs[-1] * (1 + 1e-10) and fs[0] > 0:
                    stagnation += 1
                else:
                    stagnation = 0
                
                if stagnation > 10:
                    break
            
            # If we made good progress, reduce budget for next restart
            if best_f < 1e-6:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
