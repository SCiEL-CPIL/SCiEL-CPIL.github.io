"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B with adaptive restarts and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # Counting wrapper
        call_count = [0]
        def counting_func(x):
            call_count[0] += 1
            return func(x)

        best_x = None
        best_f = np.inf

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        start_points = []

        # Center
        start_points.append(np.zeros(dim))
        # Random points
        for _ in range(8):
            start_points.append(rng.uniform(-4, 4, dim))
        # Points near corners of the feasible region
        for _ in range(4):
            start_points.append(rng.uniform(-4.5, 4.5, dim))

        # Shuffle and limit to budget
        np.random.default_rng(123).shuffle(start_points)

        for x0 in start_points:
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break

            # Clip initial point
            x0 = np.clip(x0, -5.0, 5.0)

            try:
                result = minimize(
                    counting_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break

        # If we still have budget, do a final polish with more iterations
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            try:
                result = minimize(
                    counting_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'maxfun': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Final fallback if nothing worked
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count[0] += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
