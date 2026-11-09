# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer with strict budget and time guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        # Custom exception to stop optimization early
        class BudgetExceeded(Exception):
            pass

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise BudgetExceeded()
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise BudgetExceeded()
            
            # Clip input to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Multi-start strategy
        # Number of starts depends on budget. 
        # Nelder-Mead in 5D takes roughly 50-100 calls to converge.
        # With budget 500, we can afford ~5-10 starts.
        num_starts = 5
        
        for i in range(num_starts):
            if calls_made[0] >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Random start in [-4, 4]
            x0 = rng.uniform(-4, 4, n)
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100, 
                        'xatol': 1e-4, 
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except BudgetExceeded:
                break
            except Exception:
                # Catch any other numerical issues
                continue

        return best_x, best_f
# EVOLVE-BLOCK-END
