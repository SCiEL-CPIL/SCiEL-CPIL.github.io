"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial.distance import cdist

class Optimizer:
    """
    Bayesian Optimizer for Büche-Rastrigin f4 on [-5,5]^5.
    
    Combines:
    1. Space-filling initialization (Sobol/Latin Hypercube) for good global coverage.
    2. Gaussian Process (GP) surrogate with Expected Improvement (EI) acquisition.
    3. Local (1+1)-ES polish for final precision.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # Data storage
        X_obs = []
        y_obs = []
        best_x = None
        best_f = float('inf')
        budget_used = 0

        def update_best(x, f_val):
            nonlocal best_x, best_f
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        def evaluate(x):
            nonlocal budget_used
            x = np.clip(x, -5, 5)
            f_val = func(x)
            budget_used += 1
            X_obs.append(x)
            y_obs.append(f_val)
            update_best(x, f_val)
            return f_val

        # Phase 1: Initialization
        # Use a simple space-filling design. Sobol is ideal but requires scipy.stats.qmc.
        # Let's use a mix of random and grid-like points to ensure coverage.
        n_init = 50
        init_points = []
        
        # Add origin and axis-aligned points
        init_points.append(np.zeros(dim))
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = 1.0
            init_points.append(pt)
            pt[i] = -1.0
            init_points.append(pt)
            
        # Fill rest with random points, ensuring spread
        while len(init_points) < n_init:
            pt = rng.uniform(-5, 5, dim)
            # Simple rejection if too close to existing points (optional, might be slow)
            init_points.append(pt)
            
        init_points = np.array(init_points[:n_init])
        
        for pt in init_points:
            if budget_used >= budget or not time_ok():
                break
            evaluate(pt)

        if budget_used >= budget or not time_ok():
            return best_x, best_f

        # Phase 2: Bayesian Optimization
        # GP Hyperparameters
        # Normalize y for stability
        # We will fit GP on normalized y, then scale EI
        
        # EI Optimization settings
        n_restarts_ei = 10
        
        while budget_used < budget and time_ok():
            # Fit GP
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # Normalize y
            y_mean = np.mean(y)
            y_std = np.std(y)
            if y_std < 1e-8:
                y_std = 1.0
            y_norm = (y - y_mean) / y_std
            
            # GP Kernel: Matern-5/2
            # K(x, x') = (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
            # where d = ||x - x'|| / l
            l = 1.0  # Length scale
            sigma_f = 1.0  # Signal variance
            sigma_n = 1e-4  # Noise variance
            
            # Compute kernel matrix
            # D = cdist(X, X) / l
            D = cdist(X, X) / l
            K = (1 + np.sqrt(5)*D + (5/3)*D**2) * np.exp(-np.sqrt(5)*D)
            K += sigma_n * np.eye(len(X))
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter if Cholesky fails
                K += 1e-6 * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                except:
                    break # GP failed, break loop
            
            # Solve for alpha
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            
            # Define EI acquisition function
            def neg_ei(x_candidate):
                x_c = np.array([x_candidate])
                # Kernel vector k*
                D_star = cdist(x_c, X) / l
                k_star = (1 + np.sqrt(5)*D_star + (5/3)*D_star**2) * np.exp(-np.sqrt(5)*D_star)
                k_star = k_star.flatten()
                
                # Mean and Variance
                mu_star = k_star @ alpha
                v = np.linalg.solve(L, k_star)
                var_star = sigma_f**2 - k_star @ v
                var_star = max(var_star, 1e-10)
                std_star = np.sqrt(var_star)
                
                # Best observed normalized value
                y_best_norm = -np.max(y_norm) # Since we minimize, best is lowest y. 
                # Wait, EI is for minimization. 
                # EI(x) = (y_best - mu(x)) * Phi((y_best - mu(x))/sigma) + sigma * phi(...)
                # Here y_norm is normalized such that lower is better.
                # y_best_norm is the minimum of y_norm.
                
                improvement = y_best_norm - mu_star
                z = improvement / std_star
                ei = improvement * norm.cdf(z) + std_star * norm.pdf(z)
                
                # Scale back to original scale? 
                # Actually, EI is in normalized space. 
                # We just need to maximize it. 
                return -ei
            
            # Optimize EI
            best_ei_val = -np.inf
            best_x_ei = None
            
            # Start points: random + current best
            starts = [best_x.copy()]
            for _ in range(n_restarts_ei - 1):
                starts.append(rng.uniform(-5, 5, dim))
                
            for start_pt in starts:
                res = minimize(neg_ei, start_pt, method='L-BFGS-B', bounds=[(-5, 5)]*dim, options={'maxiter': 50})
                if res.success:
                    ei_val = -res.fun
                    if ei_val > best_ei_val:
                        best_ei_val = ei_val
                        best_x_ei = res.x.copy()
                        
            if best_x_ei is None:
                # Fallback to random
                best_x_ei = rng.uniform(-5, 5, dim)
            
            # Evaluate the best EI point
            evaluate(best_x_ei)
            
            # Optional: If we have very little budget left, break to polish
            if budget - budget_used < 20:
                break

        # Phase 3: Local Polish
        # Use remaining budget for (1+1)-ES around best_x
        remaining = budget - budget_used
        if remaining > 5 and best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            min_sigma = 1e-6
            max_sigma = 1.0
            window_size = 5
            success_window = []
            
            for _ in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    success_window.append(1)
                else:
                    success_window.append(0)
                
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.3:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    sigma = max(min_sigma, min(sigma, max_sigma))

        return best_x, best_f
# EVOLVE-BLOCK-END
