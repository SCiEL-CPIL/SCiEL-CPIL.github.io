import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        calls_used = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls_used
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e10
            calls_used += 1
            if np.isnan(fx) or np.isinf(fx):
                fx = 1e10
            return fx

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 1: Random Initialization ---
        # Use a small portion of the budget to scatter points
        # This helps identify the general region of the minimum for shifted functions
        n_init = 15
        if self.budget > n_init:
            X_init = rng.uniform(-4.0, 4.0, (n_init, self.dim))
            for x in X_init:
                fx = safe_func(x)
                update_best(x, fx)
        
        if time.time() > deadline:
            return best_x, best_f

        # --- Phase 2: CMA-ES with Restarts ---
        # Parameters for CMA-ES
        # lambda = 9 is standard for dim=5
        lam = 9
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 1) - np.arange(mu)
        w /= w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        # Adaptation rates
        cc = (4 + mu / self.dim) / (self.dim + 4 + 2 * mu / self.dim)
        cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        cmua = 2 * (c1 - 2 / (self.dim + 1))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (self.dim + 1)) - 1) + cs
        chi_n = np.sqrt(self.dim) * (1 + 1/(4*self.dim) - 1/(21*self.dim**2))
        
        # State variables
        x_mean = best_x.copy() if best_x is not None else np.zeros(self.dim)
        sigma = 1.0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # Restart logic variables
        stagnation_counter = 0
        max_stagnation = 10 # Generations without improvement
        restart_count = 0
        max_restarts = 10 # Allow many restarts to explore different basins
        
        # Main loop
        while calls_used + lam <= self.budget and time.time() < deadline:
            # Sample population
            try:
                # Use Cholesky for sampling efficiency and stability
                # Ensure C is positive definite
                C = (C + C.T) / 2
                # Add small regularization if needed
                D, V = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                
                Z = rng.standard_normal((lam, self.dim))
                Y = (Z * D) @ V.T
                x_pop = x_mean + sigma * Y
                x_pop = np.clip(x_pop, -5.0, 5.0)
                
                # Evaluate population
                f_pop = np.array([safe_func(x) for x in x_pop])
                
                if np.any(np.isnan(f_pop)):
                    break
                    
                # Update global best
                min_idx = np.argmin(f_pop)
                if f_pop[min_idx] < best_f:
                    update_best(x_pop[min_idx], f_pop[min_idx])
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                # Sort by fitness
                o = np.argsort(f_pop)
                x_sorted = x_pop[o]
                f_sorted = f_pop[o]
                
                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = np.sum(w[:, None] * x_sorted[:mu], axis=0)
                
                # Update evolution paths
                # y = (x_mean - x_mean_old) / sigma
                y = (x_mean - x_mean_old) / sigma
                
                # Transform y to C^-1/2 space for ps update
                # C^-1/2 = V diag(1/D) V^T
                # y_trans = C^-1/2 y
                D_inv = 1 / np.sqrt(np.maximum(D, 1e-12))
                y_trans = (y @ V) * D_inv @ V.T
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y_trans
                norm_ps = np.linalg.norm(ps)
                
                # h_sigma condition
                # Use a simplified check to avoid complex history tracking in this short budget
                # Standard condition: ||ps|| / sqrt(1 - (1-cs)^(2*t)) < (1.4 + 2/(n+1)) * chi_n
                # For short runs, we can approximate the denominator or just use the norm
                # Let's use the standard formula with t = calls_used // lam
                t_gen = max(1, calls_used // lam)
                denom = np.sqrt(1 - (1 - cs) ** (2 * t_gen))
                h_sig = (norm_ps / denom) < (1.4 + 2 / (self.dim + 1)) * chi_n
                
                pc = (1 - cc) * pc + h_sig * np.sqrt(cc * (2 - cc) * mu_eff) * y
                
                # Update covariance
                delta_h = (h_sig - 1 + cc) * np.outer(pc, pc)
                C = (1 + c1 * cmua - c1) * C + c1 * (np.outer(x_mean - x_mean_old, x_mean - x_mean_old) / sigma**2 + delta_h)
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                D, V = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
                C = (V * D) @ V.T
                
                # Update sigma
                sigma *= np.exp((cs / d_sigma) * (norm_ps / chi_n - 1))
                
                # Check for stagnation and restart
                if stagnation_counter > max_stagnation:
                    if restart_count < max_restarts:
                        restart_count += 1
                        # Decide restart strategy
                        # 50% chance to restart from global best (local search)
                        # 50% chance to restart from random (global search)
                        if rng.random() < 0.5 and best_x is not None:
                            x_mean = best_x.copy()
                            # Add small noise to avoid exact repetition if stuck
                            x_mean = x_mean + rng.normal(0, 0.1, self.dim)
                        else:
                            x_mean = rng.uniform(-4.0, 4.0, self.dim)
                        
                        x_mean = np.clip(x_mean, -5.0, 5.0)
                        sigma = 1.0 # Reset to moderate step size
                        C = np.eye(self.dim)
                        pc = np.zeros(self.dim)
                        ps = np.zeros(self.dim)
                        stagnation_counter = 0
                    else:
                        # If max restarts reached, just continue with current settings
                        # Maybe shrink sigma to try to squeeze out more improvement
                        sigma *= 0.9
                        stagnation_counter = 0
                        
            except Exception:
                break

        # --- Phase 3: Local Polish (if budget remains) ---
        # A simple (1+1)-ES or coordinate descent can help if we have calls left
        remaining = self.budget - calls_used
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            stall = 0
            max_stall = 10
            
            while calls_used < self.budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.05
                    stall = 0
                    update_best(x, fx)
                else:
                    sigma *= 0.9
                    stall += 1
                    if stall > max_stall:
                        break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = 0.0

        return best_x, best_f
