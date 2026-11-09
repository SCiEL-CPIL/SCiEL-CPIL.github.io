import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit

        # CMA-ES Parameters
        # Smaller lambda for more generations in fixed budget
        lam = 6 
        mu = lam // 2
        
        # Weights
        # Log-weights
        weights = np.log(np.arange(lam, 0, -1))
        weights -= weights.sum()
        weights = weights[:mu]
        weights /= weights.sum()
        
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial state
        # Start at center
        mean = np.zeros(dim)
        # Initial sigma: Domain is [-5,5], optimum in [-4,4].
        # 1.5 allows covering most of the domain in ~3-4 steps if aligned, 
        # but CMA-ES adapts.
        sigma = 1.5
        
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        rng = np.random.default_rng(42)
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Helper to update inverse square root
        def update_c_inv_sqrt():
            nonlocal C, C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        update_c_inv_sqrt()
        
        no_improve = 0
        stagnation_limit = 20 # Generations
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            z = rng.standard_normal((lam, dim))
            # Transform to x
            # y = C_inv_sqrt @ z
            # x = mean + sigma * y
            y = z @ C_inv_sqrt.T
            x_pop = mean[None, :] + sigma * y
            
            # Evaluate
            f_vals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    f_vals[i] = best_f
                    continue
                # Clip to box [-5, 5]
                x_i = np.clip(x_pop[i], -5, 5)
                f_vals[i] = func(x_i)
                calls_made += 1
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_x = x_i.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
            # Sort
            order = np.argsort(f_vals)
            x_sorted = x_pop[order]
            
            # Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update paths and covariance
            if np.linalg.norm(mean - mean_old) > 1e-12:
                # p_sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ (mean - mean_old))
                
                # h_sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                # Approximation for chi-square mean, standard CMA-ES uses E[||N(0, I)||]
                # For dim=5, E[||N||] ~ 2.13. 
                # Standard formula: (1.4 + 2/(dim+1)) * sqrt((dim+1)/(dim-1)) * sqrt(1 - (1-cs)^(2*(gen+1)))
                # We use a simplified check for efficiency or standard formula
                # Let's use the standard threshold
                h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - cs)**(2 * (calls_made/lam + 1)))) < \
                          (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
                
                # p_c
                p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) / sigma * (mean - mean_old)
                
                # Delta h_sigma
                delta_h_sigma = 1.0 if h_sigma else 0.0
                
                # Rank-1 and Rank-mu update
                C = (1 + c1 * delta_h_sigma + cmu) * C - c1 * np.outer(p_c, p_c)
                
                # Rank-mu term
                # y_i = C_inv_sqrt @ (x_i - mean_old)
                y_selected = x_sorted[:mu] - mean_old[None, :]
                y_selected = y_selected @ C_inv_sqrt.T
                # Sum of weighted outer products
                rank_mu = np.sum(weights[:, None, None] * y_selected[:, :, None] * y_selected[:, None, :], axis=0)
                C -= cmu * rank_mu
                
                # Ensure positive definiteness
                update_c_inv_sqrt()
                
                # Adapt sigma
                # CMA-ES sigma adaptation
                # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
                chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                sigma = sigma * np.exp(cs / (dim + 3) * (norm_p_sigma / chi_n - 1))
                
                # Clip sigma
                sigma = np.clip(sigma, 1e-6, 10.0)
                
            else:
                # If mean didn't change, still adapt sigma slightly?
                # Standard CMA-ES skips if norm is 0, but sigma should decay if no progress?
                # Actually, if mean doesn't change, p_sigma decay handles it.
                pass
            
            # Stagnation restart
            if no_improve >= stagnation_limit:
                # Reset to best
                mean = best_x.copy()
                sigma = max(sigma * 1.5, 0.5)
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve = 0
                update_c_inv_sqrt()

        return best_x, best_f
