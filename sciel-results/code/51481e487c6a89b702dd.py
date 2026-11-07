"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with CMA-like step-size adaptation, stagnation restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline

        # Phase 1: broad random sampling to get a good starting point
        x_best = np.zeros(self.dim)
        fx_best = func(x_best)
        if not time_ok():
            return x_best, fx_best

        # Try a few random points to find a better starting point
        for _ in range(10):
            if not time_ok():
                break
            x_try = rng.uniform(-5, 5, size=self.dim)
            fx_try = func(x_try)
            if fx_try < fx_best:
                x_best, fx_best = x_try, fx_try

        # Phase 2: (1+1)-ES with adaptive step size
        x = x_best.copy()
        fx = fx_best
        sigma = 1.5
        min_sigma = 1e-6
        max_sigma = 10.0
        no_improve = 0
        best_f = fx
        restarts = 0
        max_restarts = 5

        for _ in range(self.budget - 11):
            if not time_ok():
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                if fy < best_f:
                    best_f = fy
                no_improve = 0
                sigma = min(sigma * 1.1, max_sigma)
            else:
                no_improve += 1
                sigma = max(sigma * 0.95, min_sigma)

            # Stagnation: restart from best point with larger sigma
            if no_improve >= 40:
                no_improve = 0
                restarts += 1
                if restarts > max_restarts:
                    break
                x = best_x.copy()
                fx = best_f
                sigma = min(sigma * 3.0, max_sigma)

            # Track best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
