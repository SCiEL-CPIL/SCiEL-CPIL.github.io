"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 300s limit
        calls = [0]
        best_x = None
        best_f = np.inf

        def counted_func(x):
            if calls[0] >= self.budget:
                return best_f
            calls[0] += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Generate diverse starting points
        n_starts = 20
        starts = []

        # Center
        starts.append(np.zeros(self.dim))

        # Random points in the box
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, self.dim))

        for start in starts:
            if time.time() > deadline:
                break
            if calls[0] >= self.budget:
                break

            # Check if we've already used all budget
            if calls[0] >= self.budget:
                break

            try:
                res = minimize(
                    counted_func,
                    start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfun': max(1, self.budget - calls[0]), 'gtol': 1e-8}
                )
            except Exception:
                pass

            if time.time() > deadline:
                break
            if calls[0] >= self.budget:
                break

        # If we still have budget left, do a few more random evaluations
        while calls[0] < self.budget and time.time() < deadline:
            x = rng.uniform(-5, 5, self.dim)
            calls[0] += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
