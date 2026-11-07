"""Rewritten optimizer: multi-start L-BFGS-B with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B for BBOB-style f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def tracked_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=float).ravel()
            calls_made += 1
            return func(x)

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Corners of the box
        for corner in [(np.array([-5.0]*dim), np.array([5.0]*dim))]:
            starts.append(corner[0])
            starts.append(corner[1])
        # Random points
        n_random = min(20, max(0, budget - 10))
        for _ in range(n_random):
            starts.append(rng.uniform(-5, 5, dim))

        bounds = [(-5.0, 5.0)] * dim

        for x0 in starts:
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-8}
                )
                fx = res.fun
                if fx < best_f:
                    best_f = fx
                    best_x = res.x.copy()
            except Exception:
                pass

            if calls_made >= budget:
                break

        # If we still have budget and time, do a fine-tune around best
        if best_x is not None and calls_made < budget and time.time() < deadline:
            try:
                res2 = minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        # Ensure at least one call was made
        if best_x is None:
            x0 = np.zeros(dim)
            try:
                fx = tracked_func(x0)
                best_x = x0
                best_f = fx
            except Exception:
                best_x = np.zeros(dim)
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
