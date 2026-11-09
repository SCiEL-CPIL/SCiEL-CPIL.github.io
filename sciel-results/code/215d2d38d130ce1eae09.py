"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and global best tracking."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Initial exploration: evaluate several diverse points
        # to get a good starting point for the (1+1)-ES
        initial_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
            np.array([1.0, -1.0, 1.0, -1.0, 1.0]) * 2.0,
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0]) * 2.0,
        ]

        for p in initial_points:
            if calls_made >= budget or time.time() > deadline:
                break
            x = np.clip(p, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Start (1+1)-ES from the best point found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        x = best_x.copy()
        fx = best_f
        sigma = 1.5

        # (1+1)-ES with adaptive sigma
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept
                x, fx = y, fy
                # Increase sigma
                sigma *= 1.3
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Reject
                # Decrease sigma
                sigma *= 0.85

            # Clamp sigma
            sigma = max(1e-5, min(sigma, 10.0))

            # If we've converged (sigma very small), try a restart
            if sigma < 1e-4 and calls_made < budget - 10:
                # Try a few random restarts
                for _ in range(min(5, budget - calls_made)):
                    if time.time() > deadline:
                        break
                    xr = rng.uniform(-4, 4, dim)
                    fr = func(xr)
                    calls_made += 1
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
                        x = xr.copy()
                        fx = fr
                        sigma = 1.5
                        break
                else:
                    # No improvement, continue with current x
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
