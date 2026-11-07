"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Optimizer for BBOB-style f11 (Discus) using SciPy L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def wrapped(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return np.inf
            if not np.all(np.isfinite(x)):
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = float(func(x_clipped))
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Start 1: Center
        try:
            res = minimize(wrapped, np.zeros(dim), method='L-BFGS-B', 
                           bounds=[(-5, 5)] * dim, 
                           options={'maxiter': 100, 'maxfun': budget - calls})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

        # If we have budget left, try a random start
        if calls < budget - 10 and time.time() < deadline:
            try:
                x_start = np.random.uniform(-3, 3, size=dim)
                res2 = minimize(wrapped, x_start, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * dim, 
                               options={'maxiter': 50, 'maxfun': budget - calls})
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
