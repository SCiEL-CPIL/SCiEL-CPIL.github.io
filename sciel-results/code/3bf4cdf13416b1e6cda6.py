import numpy as np
import time
from scipy.optimize import dual_annealing, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 290
            D = self.dim
            budget = self.budget
            bounds = [(-5.0, 5.0)] * D
            
            call_count = [0]
            best_x = [np.zeros(D)]
            best_f = [np.inf]

            def wrapped_func(x):
                if time.time() > deadline:
                    raise TimeoutError("Time limit exceeded")
                if call_count[0] >= budget:
                    raise ValueError("Budget exceeded")
                
                call_count[0] += 1
                f_val = float(func(x))
                if f_val < best_f[0]:
                    best_f[0] = f_val
                    best_x[0] = x.copy()
                return f_val

            # Phase 1: Global Search (Dual Annealing)
            # Use a smaller budget for this phase
            da_budget = int(budget * 0.2)
            
            try:
                # We can't easily limit DA's internal calls precisely without wrapping,
                # but we can set maxiter low.
                # DA calls func multiple times per iteration.
                # Let's just let it run until it uses up its "share" or raises exception.
                # To be safe, we catch the exception from our wrapper.
                
                # Note: DA might not respect the exact call count limit if it doesn't check between calls.
                # Our wrapper raises an exception if budget is hit.
                
                res_da = dual_annealing(
                    wrapped_func, 
                    bounds, 
                    maxiter=100, # Will be cut off by budget/time
                    initial_temp=5230.0, 
                    restart_temp_ratio=2e-5, 
                    visit=2.62, 
                    seed=0
                )
                # If DA succeeds, use its result as start for Powell
                start_x = res_da.x
            except (TimeoutError, ValueError):
                # If DA hit the limit, use the best found so far
                start_x = best_x[0]
            except Exception:
                start_x = best_x[0]

            # Phase 2: Local Refinement (Powell)
            # Use remaining budget
            remaining_budget = budget - call_count[0]
            if remaining_budget > 10:
                try:
                    res_powell = minimize(
                        wrapped_func, 
                        start_x, 
                        method='Powell',
                        bounds=bounds,
                        options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-12}
                    )
                except (TimeoutError, ValueError):
                    pass
                except Exception:
                    pass

            return (best_x[0], best_f[0])
