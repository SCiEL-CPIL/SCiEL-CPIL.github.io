import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Optimizer using Random Restarts with Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 305.0
        n = self.dim
        B = self.budget

        def time_ok():
            return (time.time() - start) < deadline

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def ev(x):
            nonlocal calls, best_x, best_f
            if calls >= B:
                return float('inf')
            calls += 1
            y = np.clip(x, -5, 5)
            f = func(y)
            if f < best_f:
                best_f = f
                best_x = y.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial Sampling
        # Sample 10 random points to find a decent starting region
        for _ in range(10):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, n)
            ev(x)

        # Phase 2: Local Refinement
        # Use Nelder-Mead from the best point found
        def run_nelder_mead(x0, max_iter=100):
            nonlocal calls, best_x, best_f
            
            def scipy_func(x):
                if not time_ok() or calls >= B:
                    return float('inf')
                return ev(x)
            
            try:
                # Nelder-Mead parameters: 
                # initial_simplex: default is x0 + 0.05*x0 + 0.00025 where x0==0
                # For [-5,5], we might want a larger initial simplex.
                # We can specify initial_simplex if needed, but default is usually okay.
                res = minimize(scipy_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'disp': False, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Refine the best point found in Phase 1
        run_nelder_mead(best_x, max_iter=200)

        # Phase 3: Random Restarts
        # If budget allows, try a few more random starts
        restarts = 0
        while calls < B - 50 and time_ok() and restarts < 5:
            x_new = rng.uniform(-5, 5, n)
            ev(x_new)
            
            # Only refine if this point is promising (close to best) or if we are stuck
            # Actually, just refine the new point
            run_nelder_mead(x_new, max_iter=100)
            restarts += 1

        return best_x, best_f
