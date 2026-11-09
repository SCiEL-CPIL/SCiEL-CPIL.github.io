import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def count_func(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Use scipy.optimize.dual_annealing as it is highly effective for global optimization
        # and handles bounds. We wrap the function to count calls.
        
        # dual_annealing might use more than 500 calls. We need to stop it.
        # We can use a custom objective that raises an exception when budget is exceeded.
        
        class BudgetExceeded(Exception):
            pass

        def bounded_func(x):
            if calls_made >= budget:
                raise BudgetExceeded
            # Clip to bounds just in case, though dual_annealing handles bounds
            x_clipped = np.clip(x, -5.0, 5.0)
            return count_func(x_clipped)

        # Parameters for dual_annealing
        # maxiter: max number of iterations of the outer SA loop
        # initial_temp: initial temperature
        # maxiter: we can set this high and rely on the budget exception
        # or estimate maxiter based on budget.
        # Each SA iteration involves a local search which uses many function calls.
        # A typical local search might use 50-100 calls.
        # So 500 budget / 100 calls/iter = ~5 SA iterations.
        # Let's set maxiter to a value that allows a few restarts.
        
        try:
            result = scipy.optimize.dual_annealing(
                bounded_func,
                bounds=[(-5.0, 5.0)] * dim,
                maxiter=20, # Allow up to 20 SA iterations, but budget will likely stop it earlier
                initial_temp=5.0,
                restart_temp_ratio=2.0,
                visit=2.62,
                maxvisit=15,
                seed=42
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x
        except BudgetExceeded:
            pass
        except Exception:
            pass

        # If we have budget left, do a local polish with Nelder-Mead
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline:
            try:
                # Wrap again to count
                def polish_func(x):
                    nonlocal calls_made, best_x, best_f
                    if calls_made >= budget:
                        raise BudgetExceeded
                    x_clipped = np.clip(x, -5.0, 5.0)
                    fx = count_func(x_clipped)
                    return fx

                res = scipy.optimize.minimize(
                    polish_func,
                    best_x,
                    method='Nelder-Mead',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except BudgetExceeded:
                pass
            except Exception:
                pass

        return best_x, best_f
