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
        
        n = self.dim
        total_budget = self.budget
        
        # State
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        
        def wrapped_func(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError()
            calls += 1
            if calls > total_budget:
                raise BudgetExceeded()
            return func(x)
            
        class BudgetExceeded(Exception):
            pass

        # Phase 1: Multi-start Nelder-Mead
        # We divide the budget into chunks.
        # n=5, NM needs ~50-100 evals.
        # Let's do 4 starts with 100 evals each = 400 evals.
        # Remaining 100 evals for polish.
        
        n_starts = 4
        budget_per_start = int((total_budget * 0.8) / n_starts)
        
        rng = np.random.default_rng(42)
        
        for i in range(n_starts):
            # Check deadline
            if time.time() > deadline - 5:
                break
                
            # Random start point
            x0 = rng.uniform(-5, 5, n)
            
            # Run Nelder-Mead
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': budget_per_start, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, BudgetExceeded):
                break
            except Exception:
                # Catch any other errors to keep the program alive
                pass
                
        # Phase 2: Final Polish
        # If we have budget left, refine the best point.
        remaining_budget = total_budget - calls
        if remaining_budget > 10 and best_f < np.inf:
            if time.time() < deadline - 5:
                try:
                    res = minimize(
                        wrapped_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': remaining_budget, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except (TimeoutError, BudgetExceeded):
                    pass
                except Exception:
                    pass

        return best_x, best_f
