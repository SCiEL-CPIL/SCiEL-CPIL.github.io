import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        budget = self.budget
        dim = self.dim
        
        calls = [0]
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal best_x, best_f
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # 1. Global Exploration using Sobol Sequence
        # We use a Sobol sequence to ensure low-discrepancy coverage.
        # 100 points in 5 dimensions is a good start.
        n_global = 100
        # scipy.stats.qmc.Sobol is available in modern scipy. 
        # If not, we can use a simple Halton or Random. 
        # Let's stick to Random with a fixed seed for reliability if qmc is not guaranteed, 
        # but Sobol is much better for global search.
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=dim, scramble=True, seed=42)
            global_points = sampler.random(n_global)
            # Map [0, 1] to [-5, 5]
            global_points = global_points * 10.0 - 5.0
        except ImportError:
            # Fallback to random
            rng = np.random.default_rng(42)
            global_points = rng.uniform(-5, 5, size=(n_global, dim))

        for x in global_points:
            if calls[0] >= budget or time.time() > deadline:
                break
            evaluate(x)
            
        if calls[0] == 0:
            # Safety: if we couldn't even make the first call, try zero
            evaluate(np.zeros(dim))
            return best_x, best_f

        # 2. Local Refinement using Nelder-Mead
        # Start from the best point found in global exploration.
        start_x = best_x.copy()
        remaining_budget = budget - calls[0]
        
        if remaining_budget < 10:
            return best_x, best_f

        # Define the objective function for scipy
        def scipy_func(x):
            return evaluate(x)

        # Run Nelder-Mead
        # maxiter is the number of iterations, not function calls.
        # Nelder-Mead uses dim+1 points in the simplex, so roughly dim+1 calls per iteration.
        # We want to use up to remaining_budget calls.
        max_iter = max(1, int(remaining_budget / (dim + 2)))
        
        # Set options to prevent excessive time usage
        options = {
            'maxiter': max_iter,
            'maxfev': remaining_budget,
            'xatol': 1e-4,
            'fatol': 1e-4
        }
        
        try:
            res = minimize(scipy_func, start_x, method='Nelder-Mead', options=options)
            # scipy minimize might return a result, but we track best_f internally
            # We also need to check if res.x is better
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            # If minimize fails, we still have the best from global phase
            pass

        return best_x, best_f
