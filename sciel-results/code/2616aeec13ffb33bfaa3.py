"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead approach for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            v = func(x_clipped)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x_clipped.copy()
            return v

        # 1. Initial diverse sampling
        n_init = 20
        init_points = []
        init_points.append(np.zeros(dim))
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, dim))
        
        init_values = []
        for p in init_points:
            p = np.clip(p, -5, 5)
            v = wrapped_func(p)
            init_values.append(v)
        
        # Sort init points by value
        sorted_indices = np.argsort(init_values)
        
        # 2. Multi-start Nelder-Mead from top seeds
        seeds = [init_points[i] for i in sorted_indices[:5]]
        
        for seed in seeds:
            if calls >= self.budget:
                break
            if time.time() - start_time > time_limit - 10.0:
                break
                
            try:
                res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # 3. If budget remains, do more random restarts with short NM runs
        while calls < self.budget - 20:
            if time.time() - start_time > time_limit - 10.0:
                break
            
            new_p = rng.uniform(-4, 4, dim)
            v = wrapped_func(new_p)
            
            # Only refine if it's reasonably good or we have lots of budget left
            if v < best_f + 10.0: 
                 try:
                     res = minimize(wrapped_func, new_p, method='Nelder-Mead', 
                                    options={'maxiter': 30, 'xatol': 1e-5, 'fatol': 1e-8})
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
