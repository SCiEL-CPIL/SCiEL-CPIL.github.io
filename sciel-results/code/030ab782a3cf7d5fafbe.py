"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell optimizer with careful budget management."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        dim = self.dim
        budget = self.budget

        # Counting wrapper
        def wrapped(x):
            if self._call_count >= budget:
                return 1e300
            self._call_count += 1
            try:
                return float(func(x))
            except Exception:
                return 1e300

        best_x = None
        best_f = float('inf')
        
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        # Multi-start strategy: try a few random points
        # Powell's method is a direct search method, different from DE
        n_restarts = 4
        calls_per_restart = budget // n_restarts
        
        for i in range(n_restarts):
            if self._call_count >= budget:
                break
            
            # Random start point within the box
            x0 = rng.uniform(-5.0, 5.0, dim)
            
            try:
                res = minimize(
                    wrapped,
                    x0,
                    method='Powell',
                    options={'maxiter': 1000, 'maxfev': calls_per_restart}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return None
# EVOLVE-BLOCK-END
