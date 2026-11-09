import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # We will use multiple restarts.
        # Each restart uses a portion of the budget.
        # Nelder-Mead typically needs ~100-200 evals to converge in 5D.
        # Let's use 5 restarts of 100 evals each?
        # Or dynamic: run until time/budget limit.
        
        # Better: Run Nelder-Mead with a maxiter that corresponds to remaining budget.
        # But Nelder-Mead doesn't easily report "calls made" to stop early inside the loop.
        # We can wrap the function to count calls and raise an exception to stop.
        
        class BudgetExceeded(Exception):
            pass
        
        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                raise BudgetExceeded
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            calls_made += 1
            return func(x_clipped)
        
        # Initialize with a random point
        rng = np.random.default_rng(42)
        
        # Number of restarts
        # Let's try 5 restarts.
        # Each restart gets ~100 budget.
        # Total 500.
        
        for restart in range(5):
            if calls_made >= budget:
                break
            if time.time() - start_time > 290:
                break
                
            # Initial guess
            x0 = rng.uniform(-5, 5, n)
            # Simplex initial size
            # Standard Nelder-Mead uses x0 + 0.05 * x0 (or 5% of magnitude)
            # If x0 is 0, it uses 0.05.
            # Let's use a fixed initial step size of 1.0 for robustness.
            options = {
                'maxiter': 100, # Limit iterations per restart
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
            
            try:
                res = minimize(safe_func, x0, method='Nelder-Mead', options=options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except BudgetExceeded:
                pass
            except Exception:
                pass
                
        return best_x, best_f
