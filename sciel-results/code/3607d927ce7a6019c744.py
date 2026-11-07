import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        d = self.dim
        # Phase 1: Global Search (1+1)-ES
        # Use 10% of budget for global search
        global_budget = max(10, int(0.1 * self.budget))
        
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        sigma = 2.0
        for _ in range(global_budget - 1):
            if time.time() > deadline - 5: # Leave 5s for local
                break
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87
        
        # Phase 2: Local Polish with Nelder-Mead
        remaining_budget = self.budget - calls
        
        if remaining_budget > 10:
            # Define wrapper to count calls and check time
            local_calls = [0]
            def local_func(z):
                if time.time() > deadline:
                    return np.inf
                local_calls[0] += 1
                if local_calls[0] > remaining_budget:
                    return np.inf
                return func(np.clip(z, -5, 5))
            
            # Run Nelder-Mead
            # maxiter is not exactly the number of function calls, but we can limit it
            # Nelder-Mead uses ~2d calls per iteration.
            # Set maxiter to remaining_budget / (2*d)
            max_iter = remaining_budget // (2 * d)
            
            try:
                res = minimize(local_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
