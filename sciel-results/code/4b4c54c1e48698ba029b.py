"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, deadline guard, and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Best solution found so far
        best_x = None
        best_f = np.inf

        # Try multiple restarts if time permits
        # First, do a quick random sampling to find a good starting point
        n_samples = 20
        for _ in range(n_samples):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Now do local optimization from the best point
        # Use (1+1)-ES with adaptive step size
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        calls_used = n_samples

        # Adaptive (1+1)-ES
        for _ in range(self.budget - n_samples - 1):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.3
            else:
                sigma *= 0.85

            # If we've converged (very small sigma), try a restart
            if sigma < 0.01:
                # Try a few random restarts
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    xr = rng.uniform(-5, 5, self.dim)
                    fr = func(xr)
                    calls_used += 1
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
                        x = best_x.copy()
                        fx = best_f
                        sigma = 1.0
                        break
                else:
                    # No improvement from restarts, break
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
