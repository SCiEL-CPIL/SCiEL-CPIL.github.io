import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0

        def try_func(x):
            nonlocal evals_used, best_f, best_x
            if evals_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f_val = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(f_val):
                return np.inf
            evals_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Broad Exploration with Sobol Sequence
        n_explore = min(100, budget)
        if n_explore > 0:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            n_sobol = 128
            sobol_points = sampler.random(n_sobol)
            sobol_points = sobol_points * 10.0 - 5.0
            
            sobol_values = []
            for i in range(n_explore):
                if evals_used >= budget or time.time() > deadline:
                    break
                x = sobol_points[i]
                f_val = try_func(x)
                sobol_values.append((f_val, x))
            
            if sobol_values:
                # Sort by function value
                sobol_values.sort(key=lambda item: item[0])
                # best_x and best_f are already updated in try_func
                # We keep the list for restarts
                start_points = [item[1] for item in sobol_values[:10]]
            else:
                start_points = [best_x]

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Local Refinement with Nelder-Mead
        # We will try to refine the top few Sobol points.
        # Nelder-Mead options
        options = {
            'maxiter': 500,
            'xatol': 1e-4,
            'fatol': 1e-8,
            'adaptive': True
        }
        
        # We need to be careful with the budget.
        # Each NM run might use many calls.
        # We have ~400 calls left.
        # Let's try refining the best point first.
        
        if best_f < np.inf:
            try:
                res = minimize(
                    lambda x: try_func(x), 
                    best_x, 
                    method='Nelder-Mead', 
                    options=options
                )
                # res.fun is the value at res.x, but try_func already updated best_f
            except Exception:
                pass

        # If we still have budget, try the 2nd best start point
        if len(start_points) > 1 and evals_used < budget and time.time() < deadline:
            try:
                res = minimize(
                    lambda x: try_func(x), 
                    start_points[1], 
                    method='Nelder-Mead', 
                    options=options
                )
            except Exception:
                pass
                
        # If we still have budget, try the 3rd best start point
        if len(start_points) > 2 and evals_used < budget and time.time() < deadline:
            try:
                res = minimize(
                    lambda x: try_func(x), 
                    start_points[2], 
                    method='Nelder-Mead', 
                    options=options
                )
            except Exception:
                pass

        return best_x, best_f
