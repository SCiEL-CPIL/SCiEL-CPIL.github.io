"""Combined optimizer: tiered local search with GP-based exploration for Katsuura (f23)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm


class Optimizer:
    """
    Optimizer for BBOB-style Katsuura function (f23).
    
    Strategy:
    1. Smart initial sampling: Mix of origin, local Gaussian, and global uniform points.
    2. GP-guided exploration: Use Gaussian Process surrogate with Expected Improvement
       to intelligently select next points, balancing exploration and exploitation.
    3. Tiered local search: Nelder-Mead refinement on promising points found by GP.
    4. Final high-precision polish on the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0  # Safety margin before 330s kill
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')
        
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() - start_time < time_limit - 5.0

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if not time_ok():
                return best_f
            x_clipped = np.clip(x, -5, 5)
            v = func(x_clipped)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x_clipped.copy()
            return v

        # 1. Initial Diverse Sampling
        init_points = []
        
        # Priority 1: The origin
        init_points.append(np.zeros(dim))
        
        # Priority 2: Local Gaussian sampling
        for _ in range(10):
            p = rng.normal(0, 0.5, dim)
            init_points.append(np.clip(p, -5, 5))
            
        for _ in range(10):
            p = rng.normal(0, 1.5, dim)
            init_points.append(np.clip(p, -5, 5))
            
        # Priority 3: Uniform sampling for global exploration
        for _ in range(15):
            p = rng.uniform(-4, 4, dim)
            init_points.append(p)
        
        # Evaluate initial points
        init_values = []
        for p in init_points:
            v = wrapped_func(p)
            init_values.append(v)
        
        # Sort indices by function value (ascending)
        sorted_indices = np.argsort(init_values)
        
        # 2. Tiered Local Search on top candidates
        # Tier 1: Deep search on the top 3 candidates
        top_k_tier1 = 3
        if calls < self.budget - 100 and time_ok():
            for i in range(min(top_k_tier1, len(sorted_indices))):
                seed_idx = sorted_indices[i]
                if not time_ok():
                    break
                seed = init_points[seed_idx].copy()
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 50, 
                                       'xatol': 1e-6, 
                                       'fatol': 1e-8, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Tier 2: Medium search on the next 5 candidates
        top_k_tier2 = 5
        if calls < self.budget - 60 and time_ok():
            for i in range(top_k_tier1, min(top_k_tier1 + top_k_tier2, len(sorted_indices))):
                seed_idx = sorted_indices[i]
                if not time_ok():
                    break
                seed = init_points[seed_idx].copy()
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 25, 
                                       'xatol': 1e-5, 
                                       'fatol': 1e-7, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # 3. GP-guided exploration phase
        # Collect all evaluated points so far
        X_eval = np.array([p for p in init_points])
        y_eval = np.array(init_values)
        
        # Build GP model
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=5, random_state=42)
        
        # Fit GP on initial data
        if len(X_eval) >= 2:
            gp.fit(X_eval, y_eval)
        
        # GP-based exploration loop
        gp_iterations = 0
        max_gp_iterations = 150
        
        while calls < self.budget - 30 and time_ok() and gp_iterations < max_gp_iterations:
            # Generate candidate points for acquisition
            n_candidates = 100
            X_cand = rng.uniform(-4.5, 4.5, size=(n_candidates, dim))
            
            # Predict mean and std
            mu, sigma = gp.predict(X_cand, return_std=True)
            
            # Expected Improvement acquisition
            current_best = best_f
            sigma = np.maximum(sigma, 1e-10)
            Z = (current_best - mu) / sigma
            EI = (current_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            
            # Select point with highest EI
            best_idx = np.argmax(EI)
            x_next = X_cand[best_idx]
            
            # Evaluate
            f_next = wrapped_func(x_next)
            
            # Update GP
            X_eval = np.vstack([X_eval, x_next.reshape(1, -1)])
            y_eval = np.append(y_eval, f_next)
            
            # Refit GP periodically to avoid computational overhead
            if gp_iterations % 10 == 0 or gp_iterations == max_gp_iterations - 1:
                try:
                    gp.fit(X_eval, y_eval)
                except Exception:
                    pass
            
            gp_iterations += 1
            
            # If we found a significantly better point, do a quick local refinement
            if f_next < best_f - 0.1:
                try:
                    res = minimize(wrapped_func, x_next.copy(), method='Nelder-Mead', 
                                   options={
                                       'maxiter': 20, 
                                       'xatol': 1e-5, 
                                       'fatol': 1e-7, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                        # Update GP with refined result
                        X_eval = np.vstack([X_eval, best_x.reshape(1, -1)])
                        y_eval = np.append(y_eval, best_f)
                        gp.fit(X_eval, y_eval)
                except Exception:
                    pass

        # 4. Final high-precision polish
        if calls < self.budget - 5 and best_x is not None and time_ok():
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={
                                   'maxiter': 100, 
                                   'xatol': 1e-10, 
                                   'fatol': 1e-12, 
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        # Additional restarts if budget remains
        if calls < self.budget - 15 and best_x is not None and time_ok():
            for _ in range(3):
                if not time_ok():
                    break
                if calls >= self.budget - 15:
                    break
                perturbation = rng.normal(0, 0.3, dim)
                seed = np.clip(best_x + perturbation, -5, 5)
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 15, 
                                       'xatol': 1e-5, 
                                       'fatol': 1e-7, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
