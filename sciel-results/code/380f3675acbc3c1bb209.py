"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class BudgetExceeded(Exception):
    pass

class Optimizer:
    """Multi-start L-BFGS-B optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        calls_used = 0
        max_calls = self.budget
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls_used
            if calls_used >= max_calls:
                raise BudgetExceeded()
            if time.time() > deadline:
                raise BudgetExceeded()
            calls_used += 1
            return func(x)

        rng = np.random.default_rng(42)
        bounds = [(-5.0, 5.0)] * self.dim

        # Heuristic starting points
        heuristic_starts = [
            np.ones(self.dim),
            -np.ones(self.dim),
            np.zeros(self.dim)
        ]
        
        # Multi-start strategy
        # L-BFGS-B is efficient for smooth functions.
        # We perform multiple restarts to find the basin.
        n_restarts = 20
        
        for i in range(n_restarts):
            if time.time() > deadline or calls_used >= max_calls:
                break
                
            if i < len(heuristic_starts):
                x0 = heuristic_starts[i]
            else:
                x0 = rng.uniform(-5, 5, self.dim)
            
            try:
                # Use L-BFGS-B with bounds
                # maxiter is set high enough to allow convergence, 
                # but BudgetExceeded will stop it early if budget runs out.
                res = minimize(safe_func, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-8})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except BudgetExceeded:
                # Budget ran out during this run
                break
            except Exception:
                # Ignore other exceptions to keep going if possible
                continue

        # Final polish if we have budget left
        if best_x is not None and calls_used < max_calls and time.time() < deadline:
            try:
                # Run a tighter optimization from the best point
                res_final = minimize(safe_func, best_x, method='L-BFGS-B', bounds=bounds,
                                     options={'maxiter': 200, 'ftol': 1e-12})
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = res_final.x.copy()
            except BudgetExceeded:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
