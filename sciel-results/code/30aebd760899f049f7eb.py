"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-and-optimize: local quadratic model + CMA-like adaptive step + restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s kill

        def safe_call(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return np.inf

        # Global bookkeeping
        best_x = None
        best_f = np.inf
        calls_made = 0

        def track_call(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Global sampling
        n_global = 10
        points = []
        rng = np.random.default_rng(42)
        for _ in range(n_global):
            x = rng.uniform(-4, 4, dim)
            f = safe_call(x)
            track_call(x, f)
            points.append((f, x))
            
        points.sort(key=lambda t: t[0])

        # 2. Local optimization using Powell
        # We will try to optimize from the top 3 best points
        for i in range(min(3, len(points))):
            if calls_made >= budget or time.time() > deadline:
                break
                
            x0 = points[i][1]
            
            # Define a function for scipy that tracks calls
            def objective(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return np.inf
                x = np.clip(x, -5, 5)
                f = safe_call(x)
                track_call(x, f)
                return f
                
            try:
                # Powell is robust for smooth functions with ridges
                res = minimize(objective, x0, method='Powell', options={'maxiter': 100, 'disp': False})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
