import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        rng = np.random.default_rng(12345)
        
        calls_used = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls_used
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_used += 1
            if np.isnan(fx) or np.isinf(fx):
                fx = 1e10
            return fx

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Random Search Initialization (15% budget)
        # This helps with multimodal functions like Rastrigin and Gallagher
        n_init = max(10, int(0.15 * self.budget))
        X_init = rng.uniform(-4.0, 4.0, (n_init, self.dim))
        f_init = np.array([safe_func(x) for x in X_init])
        
        # Update best
        min_idx = np.argmin(f_init)
        update_best(X_init[min_idx], f_init[min_idx])
        
        # Check time
        if time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES (75% budget)
        # CMA-ES is robust and efficient for black-box continuous optimization
        remaining_budget = self.budget - calls_used
        if remaining_budget > 20:
            # Start from the best found point
            x0 = best_x.copy()
            sigma0 = 1.0  # Initial step size, reasonable for [-5,5] box
            
            # CMA-ES parameters for dim=5
            lam = 9
            mu = lam // 2
            w = np.log(mu + 1) - np.arange(mu)
            w /= w.sum()
            mu_eff = 1 / np.sum(w ** 2)
            
            cc = (4 + mu / self.dim) / (self.dim + 4 + 2 * mu / self.dim)
            cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
            c1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
            cmua = 2 * (c1 - 2 / (self.dim + 1))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (self.dim + 1)) - 1) + cs
            chi_n = np.sqrt(self.dim) * (1 + 1/(4*self.dim) - 1/(21*self.dim**2))
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(self.dim)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            
            # Track stagnation for potential restart
            best_in_cma = best_f
            stagnation_counter = 0
            max_stagnation = 30  # Generations without improvement
            
            while calls_used + lam <= self.budget and time.time() < deadline:
                # Sample population
                # Use Cholesky or eigen decomposition for sampling
                # For small dim, eigen is fine
                try:
                    D, V = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-14))
                    Z = rng.standard_normal((lam, self.dim))
                    Y = (Z * D) @ V.T
                    x_pop = x_mean + sigma * Y
                    x_pop = np.clip(x_pop, -5.0, 5.0)
                    
                    f_pop = np.array([safe_func(x) for x in x_pop])
                    
                    # Check for NaNs
                    if np.any(np.isnan(f_pop)):
                        break
                        
                    # Update best
                    min_idx = np.argmin(f_pop)
                    if f_pop[min_idx] < best_f:
                        update_best(x_pop[min_idx], f_pop[min_idx])
                        
                    if best_f < best_in_cma:
                        best_in_cma = best_f
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                        
                    # Sort
                    o = np.argsort(f_pop)
                    x_sorted = x_pop[o]
                    
                    # Update mean
                    x_mean_old = x_mean.copy()
                    x_mean = np.sum(w[:, None] * x_sorted[:mu], axis=0)
                    
                    # Update evolution paths
                    y = (x_mean - x_mean_old) / sigma
                    # Transform y to C^-1/2 space
                    D_inv = 1 / np.sqrt(np.maximum(D, 1e-14))
                    y_trans = (y @ V) * D_inv @ V.T
                    
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y_trans
                    norm_ps = np.linalg.norm(ps)
                    h_sig = (norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2 / (self.dim + 1)) * chi_n
                    
                    pc = (1 - cc) * pc + h_sig * np.sqrt(cc * (2 - cc) * mu_eff) * y
                    
                    # Update covariance
                    delta_h = (h_sig - 1 + cc) * np.outer(pc, pc)
                    C = (1 + c1 * cmua - c1) * C + c1 * (np.outer(x_mean - x_mean_old, x_mean - x_mean_old) / sigma**2 + delta_h)
                    
                    # Ensure symmetry and positive definiteness
                    C = (C + C.T) / 2
                    D, V = np.linalg.eigh(C)
                    D = np.maximum(D, 1e-14)
                    C = (V * D) @ V.T
                    
                    # Update sigma
                    sigma *= np.exp((cs / d_sigma) * (norm_ps / chi_n - 1))
                    
                    # Stagnation restart: if no improvement for many generations,
                    # try a random restart near the best point
                    if stagnation_counter > max_stagnation:
                        # Perturb the best point
                        x_mean = best_x + rng.normal(0, 0.5, self.dim)
                        x_mean = np.clip(x_mean, -5.0, 5.0)
                        sigma = 0.5
                        C = np.eye(self.dim)
                        pc = np.zeros(self.dim)
                        ps = np.zeros(self.dim)
                        stagnation_counter = 0
                        
                except Exception:
                    break

        # Phase 3: Local Polish (10% budget)
        # Simple (1+1)-ES for final refinement
        remaining = self.budget - calls_used
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            stall = 0
            
            while calls_used < self.budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.05
                    stall = 0
                    if fx < best_f:
                        update_best(x, fx)
                else:
                    sigma *= 0.9
                    stall += 1
                    if stall > 15:
                        break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = 0.0

        return best_x, best_f
