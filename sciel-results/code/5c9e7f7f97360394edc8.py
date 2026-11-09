"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        if lam > budget // 10: # Ensure we can do at least a few generations
            lam = max(4, budget // 20)
            
        # We will run CMA-ES in segments (restarts)
        # Each segment uses a portion of the budget
        # Let's aim for about 5-10 restarts
        max_restarts = 10
        budget_per_restart = budget // max_restarts
        
        rng = np.random.default_rng(0)
        
        # Initial random start
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        budget_used = 1
        
        for r in range(max_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Check budget
            if budget_used >= budget:
                break
            
            # Initialize CMA-ES state
            # Mean starts at current best
            mean = best_x.copy()
            
            # Initial step size
            # If we are at the start, use a larger sigma. If restarting, use a smaller one to explore locally?
            # Actually, standard practice is to restart with a fresh sigma if we want to escape.
            # But since we center at best_x, a moderate sigma is good.
            sigma = 0.5 if r > 0 else 2.0
            
            # Covariance matrix C (identity)
            C = np.eye(dim)
            
            # Evolution path for step size control
            p_sigma = np.zeros(dim)
            # Evolution path for covariance adaptation
            p_c = np.zeros(dim)
            
            # Parameters
            mu = lam // 2
            weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_sigma_inv = 1.0 / c_sigma
            c_c_inv = 1.0 / c_c
            
            # Cholesky of C
            C_chol = np.linalg.cholesky(C)
            
            # Stagnation counter
            no_improve_count = 0
            stagnation_limit = 20 # Generations without improvement before restart
            
            # Inner loop: CMA-ES generations
            while budget_used < budget:
                if time.time() > deadline:
                    break
                    
                # Generate population
                # y_i ~ N(0, C)
                zs = rng.standard_normal((lam, dim))
                ys = zs @ C_chol.T
                xs = mean + sigma * ys
                xs = np.clip(xs, -5, 5)
                
                # Evaluate
                fs = np.array([func(x) for x in xs])
                budget_used += lam
                
                # Sort
                order = np.argsort(fs)
                xs_sorted = xs[order]
                fs_sorted = fs[order]
                
                # Update best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = xs_sorted[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > stagnation_limit:
                    break
                    
                # Recalculate mean
                mean_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                # Update evolution paths
                y_mean = (mean_new - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * np.sqrt(c_sigma * (2 - c_sigma)))
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_sigma * (c_sigma_inv * (np.outer(p_sigma, p_sigma) - C) + c_sigma_inv * (2 - c_sigma) * (np.outer(p_sigma, p_sigma) - C) * 0) + c_c * (2 - c_c) * (np.outer(p_c, p_c) - C) + delta_h * (np.outer(p_c, p_c) - C) + c_sigma * c_sigma_inv * (np.outer(p_sigma, p_sigma) - C) * 0)
                
                # Wait, the standard update formula is:
                # C <- (1 - c_sigma) C + c_sigma (y_mean y_mean^T + (1-h_sigma) c_sigma (2-c_sigma) (y_mean y_mean^T - C)) + ...
                # Let's use a simpler, robust update for small dim
                # C <- (1 - c_c) C + c_c (y_mean y_mean^T + (1 - h_sigma) c_c (2 - c_c) (y_mean y_mean^T - C))
                # Actually, let's stick to the standard CMA-ES update but simplified for clarity and stability
                # C = (1 - c_sigma) * C + c_sigma * (np.outer(y_mean, y_mean) + (1 - h_sigma) * c_sigma * (2 - c_sigma) * (np.outer(y_mean, y_mean) - C))
                # This is getting complex to get exactly right without a library. 
                # Given the budget is small (500), a simpler adaptation might suffice.
                # Let's use a simplified rank-one update for C based on the step.
                
                # Simplified Rank-One Update for C
                # C <- (1 - c_c) C + c_c * outer(y_mean, y_mean)
                # This ignores the rank-mu update which is crucial for high performance, 
                # but for dim=5 and small budget, rank-one might be okay. 
                # However, CMA-ES power comes from rank-mu.
                
                # Let's implement the full update carefully.
                # y_mean is the normalized step.
                # C update:
                # C <- (1 - c_sigma) C + c_sigma (y_mean y_mean^T + (1-h_sigma) c_sigma (2-c_sigma) (y_mean y_mean^T - C))
                # This is the rank-one part.
                # Plus the rank-mu part:
                # + c_c (2 - c_c) (sum_{i=1}^mu w_i y_i y_i^T - C)
                
                # Let's compute the weighted sum of outer products of the selected steps
                # y_i = (x_i - mean_old) / sigma
                # We have xs_sorted[:mu]
                # y_selected = (xs_sorted[:mu] - mean) / sigma
                y_selected = (xs_sorted[:mu] - mean) / sigma
                H = np.sum(weights[:, None, None] * y_selected[:, :, None] * y_selected[:, None, :], axis=0)
                
                # Standard CMA-ES C update:
                # C <- (1 - c_sigma) C + c_sigma (y_mean y_mean^T + (1-h_sigma) c_sigma (2-c_sigma) (y_mean y_mean^T - C)) 
                #    + c_c (2 - c_c) (H - C)
                
                # Let's re-evaluate the formula.
                # The term (1-h_sigma) c_sigma (2-c_sigma) (y_mean y_mean^T - C) is part of the rank-one update correction.
                # Actually, the standard formula is:
                # C <- (1 - c_sigma) C + c_sigma (y_mean y_mean^T + (1-h_sigma) c_sigma (2-c_sigma) (y_mean y_mean^T - C)) + c_c (2-c_c) (H - C)
                # This seems redundant. Let's use the cleaner formulation:
                # C <- (1 - c_sigma) C + c_sigma (y_mean y_mean^T + (1-h_sigma) c_sigma (2-c_sigma) (y_mean y_mean^T - C)) + c_c (2-c_c) (H - C)
                
                # Wait, the term (1-h_sigma) is for the step size control, not C.
                # The C update is:
                # C <- (1 - c_sigma) C + c_sigma (y_mean y_mean^T + (1-h_sigma) c_sigma (2-c_sigma) (y_mean y_mean^T - C)) + c_c (2-c_c) (H - C)
                
                # Let's just use the simplified rank-mu update which is more stable for small budgets:
                # C <- (1 - c_c) C + c_c * H
                # This ignores the rank-one part, which is okay for short runs.
                
                C = (1 - c_c) * C + c_c * H
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                try:
                    C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_chol = np.eye(dim)
                    
                # Update sigma
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - np.sqrt(mu_eff)))
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Update mean
                mean = mean_new
                
                # Check for convergence/stagnation in this restart
                if sigma < 1e-4:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
