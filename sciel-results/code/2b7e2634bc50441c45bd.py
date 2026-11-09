"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead based black-box optimizer for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        call_count = [0]
        max_calls = self.budget

        def wrapped_func(x):
            if call_count[0] >= max_calls:
                return 1e10
            call_count[0] += 1
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            return val

        best_x = np.zeros(self.dim)
        best_f = float('inf')

        # Phase 1: Initial Nelder-Mead from origin
        # Allocate most of the budget to the first run
        remaining_budget = max_calls
        
        # Run 1
        if remaining_budget > 0:
            options = {
                'maxfev': min(remaining_budget, 400), # Leave 100 for potential second run
                'xatol': 1e-8,
                'fatol': 1e-8
            }
            try:
                res1 = minimize(wrapped_func, best_x, method='Nelder-Mead', options=options)
                if res1.fun < best_f:
                    best_f = res1.fun
                    best_x = res1.x
            except Exception:
                pass

        # Phase 2: Refinement from the best point found
        remaining_budget = max_calls - call_count[0]
        if remaining_budget > 0 and call_count[0] < max_calls:
            options2 = {
                'maxfev': remaining_budget,
                'xatol': 1e-10,
                'fatol': 1e-10
            }
            try:
                res2 = minimize(wrapped_func, best_x, method='Nelder-Mead', options=options2)
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
