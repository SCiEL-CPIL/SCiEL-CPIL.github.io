"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Periodicity-aware CMA-ES with stagnation-triggered large-sigma restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        dim = self.dim
        budget = self.budget
        
        def check_time():
            return time.time() > deadline

        def run_cmaes(x_start, sigma_start, budget_left, stagnation_threshold=10):
            nonlocal best_x, best_f, calls_used
            
            if budget_left <= 0 or check_time():
                return
            
            # CMA-ES parameters
            lambda_pop = 10
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1]) / ((dim + 2)**2 + mu_eff))
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1/(2*dim) + 1/(36*dim**2))
            
            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            
            # Evaluate initial mean
            if budget_left < 1:
                return
            fx = func(x_mean)
            calls_used += 1
            budget_left -= 1
            if fx < best_f:
                best_f = fx
                best_x = x_mean.copy()
            
            last_improvement_call = calls_used
            local_best_f = best_f
            
            while budget_left >= lambda_pop:
                if check_time():
                    return
                
                # Sample population
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(dim)
                    
                ys = rng.standard_normal((lambda_pop, dim))
                xs = x_mean[None, :] + sigma * (ys @ L.T)
                xs = np.clip(xs, -5, 5)
                
                # Evaluate
                fs = np.array([func(x) for x in xs])
                calls_used += lambda_pop
                budget_left -= lambda_pop
                
                # Update best
                idx_min = np.argmin(fs)
                if fs[idx_min] < best_f:
                    best_f = fs[idx_min]
                    best_x = xs[idx_min].copy()
                    last_improvement_call = calls_used
                
                # Check for stagnation: if no improvement for 'stagnation_threshold' calls
                # and we have enough budget left, restart with larger sigma.
                if (calls_used - last_improvement_call >= stagnation_threshold) and (budget_left > lambda_pop * 3):
                    # Restart: center on current best, increase sigma significantly
                    # to jump out of the local basin (Rastrigin periodicity ~6.28)
                    x_mean = best_x.copy()
                    sigma = max(sigma * 2.0, 3.0)
                    C = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    last_improvement_call = calls_used
                    continue

                # Sort by fitness
                idx_sorted = np.argsort(fs)
                xs_sorted = xs[idx_sorted]
                fs_sorted = fs[idx_sorted]
                
                # Update mean
                x_new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                # Compute C_inv_sqrt
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Update step size (sigma)
                y_mean = (x_new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * y_mean
                sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update path p_c
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (C_inv_sqrt @ y_mean)
                
                # Update covariance matrix C
                y_bar = C_inv_sqrt @ y_mean
                H = np.zeros((dim, dim))
                for i in range(mu):
                    y_diff = C_inv_sqrt @ ((xs_sorted[i] - x_mean) / sigma) - y_bar
                    H += weights[i] * np.outer(y_diff, y_diff)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - c_c) * C) + c_mu * H
                C = (C + C.T) / 2
                
                x_mean = x_new_mean

            return

        # Strategy:
        # 1. Start at center (0,0,0,0,0) or random? 
        # Rastrigin minimum is at 0 (before rotation/shift). 
        # With rotation/shift, the minimum is unknown.
        # Start at 0 is a good bet for Rastrigin structure.
        # Start 1: 0, sigma=2.0
        x_start1 = np.zeros(dim)
        run_cmaes(x_start1, 2.0, self.budget // 2)
        
        if calls_used < self.budget and not check_time():
            # Start 2: Random point
            x_start2 = rng.uniform(-5, 5, dim)
            run_cmaes(x_start2, 2.0, self.budget - calls_used)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
