"""Multi-start L-BFGS-B optimizer for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.full(dim, 5.0)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        def time_left():
            return deadline - time.time() > 1.0

        def calls_left():
            return budget - calls

        # Wrap func to count calls and handle bounds
        def wrapped_func(x):
            nonlocal calls
            if calls >= budget or not time_left():
                return np.inf
            calls += 1
            try:
                val = func(np.clip(x, -5, 5))
                return val
            except Exception:
                return np.inf

        # Generate initial points
        # 1. Origin
        # 2. Random points in [-4, 4]^d
        # 3. Points near corners
        rng = np.random.default_rng(42)
        
        initial_points = []
        initial_points.append(np.zeros(dim))
        
        # Add some structured points
        for i in range(10):
            initial_points.append(rng.uniform(-4, 4, dim))
            
        # Add points near boundaries
        for i in range(5):
            p = np.zeros(dim)
            p[i % dim] = 4.0
            initial_points.append(p)
            p[i % dim] = -4.0
            initial_points.append(p)

        # Shuffle to ensure diversity in early calls if we stop early
        rng.shuffle(initial_points)

        # Run L-BFGS-B from each start
        # We allocate a portion of the budget per start, but stop if we run out
        # L-BFGS-B is efficient, so 50-100 calls per start is usually enough for local opt
        # We have 500 calls total. ~20 starts * 20 calls = 400 calls.
        
        for x0 in initial_points:
            if not time_left() or calls >= budget:
                break
            
            # Limit iterations to ensure we don't waste too many calls on one bad start
            # maxiter=50 is a safe upper bound for local refinement
            try:
                res = minimize(
                    wrapped_func, x0, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # If we have budget left, do a final high-precision polish from the best point
        if time_left() and calls < budget and best_f < np.inf:
            try:
                res = minimize(
                    wrapped_func, best_x, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
