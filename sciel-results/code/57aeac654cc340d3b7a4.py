"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Hybrid global-local optimizer for BBOB f7 (step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim

        # Phase 1: Global exploration with random restarts
        best_x = None
        best_f = np.inf
        n_global = 0
        max_global_calls = 100

        for _ in range(max_global_calls):
            if time.time() > deadline or n_global >= self.budget:
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            n_global += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: return origin if no valid evaluation
            return np.zeros(d), 0.0

        # Phase 2: Local refinement using coordinate-wise Brent optimization
        # For each coordinate, do a 1D line search around the current best
        current_x = best_x.copy()
        current_f = best_f
        remaining_budget = self.budget - n_global

        # Allocate budget for local search
        # Each coordinate gets a portion of remaining budget
        calls_per_coord = max(10, remaining_budget // (d * 3))  # 3 passes over coordinates

        for pass_num in range(3):
            for i in range(d):
                if time.time() > deadline:
                    break
                if n_global >= self.budget:
                    break

                # Define 1D objective: vary coordinate i, keep others fixed
                def obj_1d(t, i=i):
                    if n_global >= self.budget:
                        return current_f
                    x_test = current_x.copy()
                    x_test[i] = t
                    ft = func(x_test)
                    n_global += 1
                    return ft

                # Brent's method on [-5, 5] for coordinate i
                try:
                    result = minimize_scalar(obj_1d, bounds=(-5, 5), method='bounded',
                                            options={'maxiter': min(calls_per_coord, self.budget - n_global + 1)})
                    if result.fun < current_f:
                        current_x[i] = result.x
                        current_f = result.fun
                except Exception:
                    pass

            # If we've used most of the budget, break early
            if n_global > self.budget * 0.9:
                break

        return current_x, current_f
# EVOLVE-BLOCK-END
