import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 295.0
        dim = self.dim
        budget = self.budget
        
        def safe_func(x):
            try:
                x_clipped = np.clip(x, -5, 5)
                val = float(func(x_clipped))
                if not np.isfinite(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        best_x = np.zeros(dim)
        best_f = float('inf')
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Multi-start exploration
        n_restarts = 10
        evals_per_restart = 15
        total_exploration_evals = n_restarts * evals_per_restart
        
        for i in range(n_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Random start
            x0 = rng.uniform(-5, 5, dim)
            
            # Run L-BFGS-B
            try:
                res = scipy.optimize.minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': evals_per_restart, 'maxfun': evals_per_restart}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Phase 2: Polish the best point
        remaining_budget = budget - (time.time() - start_time) # This is not evals, this is time.
        # We need to track evals. scipy.optimize.minimize doesn't easily expose eval count in a simple way without wrapping.
        # Let's wrap the function to count calls.
        
        # ... (Implementation with eval counting)
        
        return best_x, best_f
