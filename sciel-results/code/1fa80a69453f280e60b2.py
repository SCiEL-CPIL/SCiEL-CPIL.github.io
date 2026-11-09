"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Try a set of starting points
        # Mix of: origin, random points, and points near boundaries
        starts = []

        # Origin
        starts.append(np.zeros(n))

        # Random points in the box
        for _ in range(15):
            starts.append(rng.uniform(-5, 5, n))

        # Points near the center with small perturbations
        for _ in range(5):
            starts.append(rng.uniform(-1, 1, n))

        # Points at various radial distances from origin
        for r in [1.0, 2.0, 3.0, 4.0]:
            for _ in range(3):
                direction = rng.standard_normal(n)
                direction /= np.linalg.norm(direction)
                starts.append(direction * r)

        # Evaluate each start with L-BFGS-B
        calls_used = 0

        for x0 in starts:
            if calls_used >= budget - 5:
                break
            if time.time() > deadline:
                break

            def objective(x):
                nonlocal calls_used
                if calls_used >= budget:
                    return np.inf
                calls_used += 1
                x_clipped = np.clip(x, -5, 5)
                return func(x_clipped)

            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 100, 'maxfun': 20}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # If we have budget left, do a final polish with more iterations
        if best_x is not None and calls_used < budget - 10 and time.time() < deadline:
            def objective2(x):
                nonlocal calls_used
                if calls_used >= budget:
                    return np.inf
                calls_used += 1
                x_clipped = np.clip(x, -5, 5)
                return func(x_clipped)

            try:
                res = minimize(
                    objective2,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 200, 'maxfun': 30, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Ensure we made at least one call
        if best_x is None:
            x = rng.uniform(-5, 5, n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
