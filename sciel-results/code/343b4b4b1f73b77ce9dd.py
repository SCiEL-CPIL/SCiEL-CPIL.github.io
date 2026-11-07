"""Optimizer for BBOB f6 (Attractive sector) using Multi-Start Nelder-Mead."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer for f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        best_x = None
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_x = x_clipped.copy()
                best_f = f
            return f

        # Phase 1: Diverse Initialization
        # Use a few random points and the origin to find a good starting basin
        rng = np.random.default_rng(0)
        init_points = []
        init_points.append(np.zeros(dim))
        for _ in range(9):
            init_points.append(rng.uniform(-4.0, 4.0, dim))
        
        init_points = np.array(init_points)
        
        init_results = []
        for x in init_points:
            if time.time() > deadline or calls >= budget:
                break
            f = eval_func(x)
            init_results.append((f, x))
        
        if not init_results:
            # Fallback if something went wrong
            x0 = np.zeros(dim)
            eval_func(x0)
            return best_x, best_f

        # Sort by fitness
        init_results.sort(key=lambda x: x[0])
        
        # Phase 2: Local Optimization with Nelder-Mead
        # We will try to optimize from the best 2 starting points if budget allows
        # Nelder-Mead in 5D is efficient.
        
        # Calculate remaining budget
        remaining_budget = budget - calls
        
        # Allocate budget to the best candidate
        # Nelder-Mead typically needs a few hundred iterations for good convergence
        # but we cap it to remaining budget.
        
        # Strategy: Use 80% of remaining budget for the best point, 20% for the second if it exists.
        # Or just use all for the best if the second is much worse.
        
        candidates_to_optimize = [init_results[0][1]]
        if len(init_results) > 1:
            # Only optimize the second best if it's close to the best
            if init_results[1][0] < init_results[0][0] + 1.0: 
                candidates_to_optimize.append(init_results[1][1])

        for i, x0 in enumerate(candidates_to_optimize):
            if time.time() > deadline or calls >= budget:
                break
            
            # Determine budget for this restart
            if i == 0 and len(candidates_to_optimize) > 1:
                alloc = int(remaining_budget * 0.7)
            else:
                alloc = budget - calls
            
            if alloc < 10:
                break
                
            # Define the objective for scipy
            def objective(z):
                return eval_func(z)
                
            # Nelder-Mead options
            # xatol and fatol control convergence
            # maxiter is the max number of function evaluations (approx)
            # Note: scipy's Nelder-Mead maxiter is not exactly 1:1 with function calls, 
            # but it's a good heuristic. We use alloc as maxiter.
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': alloc, 
                        'xatol': 1e-6, 
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                # The function `objective` calls `eval_func` which updates `best_x` and `best_f`
                # so we don't strictly need to check res.fun, but it's good practice
                if res.fun < best_f:
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
                    best_f = res.fun
            except Exception:
                # If scipy fails, just continue
                pass

        return best_x, best_f
