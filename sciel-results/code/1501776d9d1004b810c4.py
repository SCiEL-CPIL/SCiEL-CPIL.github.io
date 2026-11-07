"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead optimizer with tracking."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        # Track the best point and value found during the optimization
        def tracked_func(x):
            val = float(func(x))
            if val < self.best_f:
                self.best_f = val
                self.best_x = np.array(x, dtype=float)
            return val

        # Initialize with a random point in a smaller box to ensure we are likely in the basin of attraction
        # The minimizer is in [-4, 4]^5, so starting in [-2, 2]^5 is a reasonable heuristic.
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-2, 2, self.dim)
        
        # Evaluate initial point
        self.best_f = tracked_func(x0)
        
        # Use Nelder-Mead method
        # maxiter corresponds to the number of iterations, each consuming 1 function call.
        # We have self.budget - 1 calls left after the initial evaluation.
        max_iter = self.budget - 1
        
        try:
            res = minimize(
                tracked_func,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
        except Exception:
            # If optimization fails, we still return the best point found so far
            pass

        # Ensure best_x is not None (shouldn't happen since we evaluated x0)
        if self.best_x is None:
            self.best_x = x0
            self.best_f = float(func(x0))

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
