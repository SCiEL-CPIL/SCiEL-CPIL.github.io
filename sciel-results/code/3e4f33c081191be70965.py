"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        calls_used = 0

        # Define a wrapper to count function calls and enforce deadline
        def wrapped_func(x):
            nonlocal calls_used
            if time.perf_counter() > deadline:
                raise TimeoutError
            calls_used += 1
            return func(x)

        # Multi-start strategy: use 3 distinct starting points
        # 1. Origin (0,0,0,0,0) - often good for centered problems
        # 2. Random point 1
        # 3. Random point 2
        starts = [
            np.zeros(self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim)
        ]

        for x0 in starts:
            if time.perf_counter() > deadline - 10.0:
                break
            if calls_used >= self.budget:
                break
            
            try:
                # Nelder-Mead is robust for Rosenbrock
                # maxiter is set high, but we rely on the time/call guard
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'adaptive': True}
                )
                
                # Check if the result is valid (converged or just finished)
                # We always take the best point found by the optimizer
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except TimeoutError:
                break
            except Exception:
                # If an exception occurs (e.g. non-finite value), continue to next start
                pass

        # Final safety check: if we haven't made any calls (shouldn't happen), make one
        if calls_used == 0:
            x0 = np.zeros(self.dim)
            best_x = x0
            best_f = func(x0)
            calls_used = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
