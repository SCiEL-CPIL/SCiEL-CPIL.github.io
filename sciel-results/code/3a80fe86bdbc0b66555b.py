import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def wrapped_func(x):
            nonlocal best_x, best_f, call_count
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count >= self.budget:
                raise StopIteration("Budget exceeded")
            
            fx = func(x)
            call_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        rng = np.random.default_rng(42)
        
        # Phase 1: Multi-start local optimization
        # Use 5 starts, 50 evaluations each
        n_starts = 5
        budget_per_start = 50
        
        for i in range(n_starts):
            if call_count >= self.budget:
                break
                
            # Random start within [-5, 5]
            x0 = rng.uniform(-5, 5, self.dim)
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxfev': budget_per_start,
                        'xatol': 1e-6,
                        'fatol': 1e-6
                    }
                )
            except (TimeoutError, StopIteration):
                break
            except Exception:
                pass

        # Phase 2: Polish the best result with remaining budget
        if best_x is not None:
            remaining_budget = self.budget - call_count
            if remaining_budget > 0:
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={
                            'maxfev': remaining_budget,
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                except (TimeoutError, StopIteration):
                    pass
                except Exception:
                    pass

        return best_x, best_f
