import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290
        d = self.dim
        max_calls = self.budget
        calls_used = 0
        best_x = np.zeros(d)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def count_and_eval(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= max_calls:
                return 1e10
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Evaluate origin first
        count_and_eval(best_x)

        # Strategy: Iterative Nelder-Mead with restarts
        # Each iteration uses a chunk of the budget.
        # If no improvement, restart from a random point.
        
        chunk_size = 50
        num_restarts = 0
        max_restarts = 10
        
        while calls_used < max_calls and time.time() < deadline - 10:
            # Determine start point
            if num_restarts == 0:
                start = best_x
            else:
                # Mix of current best and random
                if rng.random() < 0.5:
                    start = best_x + rng.normal(0, 1, d)
                else:
                    start = rng.uniform(-4, 4, d)
                
            start = np.clip(start, -5, 5)
            
            # Run Nelder-Mead
            # Use a smaller maxfev to allow for more restarts
            current_budget = min(chunk_size, max_calls - calls_used)
            
            try:
                res = minimize(count_and_eval, start, method='Nelder-Mead', 
                               options={'maxfev': current_budget, 'xatol': 1e-8, 'fatol': 1e-10})
            except Exception:
                break

            # Check if we improved
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
                num_restarts = 0 # Reset restart counter on improvement
            else:
                num_restarts += 1
                
            if num_restarts > max_restarts:
                break

        # Final Polish: Use remaining budget for a high-precision run
        if calls_used < max_calls and time.time() < deadline - 5:
            remaining = max_calls - calls_used
            if remaining > 10:
                try:
                    res = minimize(count_and_eval, best_x, method='Nelder-Mead', 
                                   options={'maxfev': remaining, 'xatol': 1e-12, 'fatol': 1e-14})
                except Exception:
                    pass

        return best_x, best_f
