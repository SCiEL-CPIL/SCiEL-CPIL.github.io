import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        n = self.dim
        budget = self.budget
        
        # State
        best_x = None
        best_fx = np.inf
        
        # Helper to run L-BFGS-B with eval limit
        # L-BFGS-B doesn't have a direct eval limit, but we can check time or use a wrapper
        # However, maxiter is the main control.
        
        # Phase 1: Multi-Start L-BFGS-B
        # We will perform 5 restarts.
        # Each restart gets a budget of ~80 evals.
        # L-BFGS-B in 5D is fast. 20 iterations is usually enough for a local min.
        # Let's allocate 400 evals for 5 restarts (80 each).
        # 100 evals for final polish.
        
        rng = np.random.default_rng(0)
        
        num_restarts = 5
        evals_per_restart = 80
        final_budget = 100
        
        for i in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
                
            # Random start
            x0 = rng.uniform(-5, 5, n)
            
            # Wrapper to count evals and stop if needed?
            # SciPy minimize doesn't easily support hard eval stop without custom callback or exception.
            # We can use a counter and raise an exception, but that's messy.
            # Better to rely on maxiter.
            # In 5D, L-BFGS-B uses ~1-2 evals per iteration for the main step + line search.
            # Let's say 10 evals per iteration.
            # 80 evals -> 8 iterations.
            # This might be too few to converge.
            # Let's increase restarts to 10, and reduce per-restart budget?
            # Or just use maxiter=15.
            
            try:
                res = minimize(
                    func, x0, method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 15, 'ftol': 1e-6, 'gtol': 1e-6}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: Final Polish
        if best_x is not None and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    func, best_x, method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # If no result was found (should not happen), return a random point
        if best_x is None:
            best_x = np.zeros(n)
            best_fx = func(best_x)

        return best_x, best_fx
