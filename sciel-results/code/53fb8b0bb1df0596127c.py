"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        calls = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def count_func(x):
            nonlocal calls
            if calls >= self.budget:
                # Return a very large value to discourage further steps
                # or raise an exception? Returning large value is safer for NM
                return 1e10
            calls += 1
            return func(x)

        rng = np.random.default_rng(123)

        # Strategy: 2 independent starts, then refine best
        # Budget allocation: 40% for start 1, 40% for start 2, 20% for refinement
        # Since NM is fast, we can just let it run until budget or convergence.
        
        # Start 1
        x0 = rng.uniform(-5, 5, self.dim)
        # Ensure x0 is not too close to 0 to get a reasonable default simplex size
        # If x0 is near 0, shift it slightly
        if np.all(np.abs(x0) < 1.0):
            x0 += 2.0
            
        try:
            res1 = opt.minimize(count_func, x0, method='Nelder-Mead',
                                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res1.fun < best_f:
                best_f = res1.fun
                best_x = res1.x
        except Exception:
            pass

        if time.time() > deadline or calls >= self.budget:
            return best_x, best_f

        # Start 2
        x0_2 = rng.uniform(-5, 5, self.dim)
        if np.all(np.abs(x0_2) < 1.0):
            x0_2 += 2.0
            
        try:
            res2 = opt.minimize(count_func, x0_2, method='Nelder-Mead',
                                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res2.fun < best_f:
                best_f = res2.fun
                best_x = res2.x
        except Exception:
            pass

        if time.time() > deadline or calls >= self.budget:
            return best_x, best_f

        # Refinement from best point
        try:
            res3 = opt.minimize(count_func, best_x, method='Nelder-Mead',
                                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res3.fun < best_f:
                best_f = res3.fun
                best_x = res3.x
        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
