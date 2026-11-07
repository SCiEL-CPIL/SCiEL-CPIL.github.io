"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and periodic restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # Initial exploration: try a few random starts, pick the best
        n_starts = min(5, budget // 100)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        for _ in range(n_starts - 1):
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            calls += 1
            if fx_try < best_f:
                best_f = fx_try
                best_x = x_try.copy()

        # Main (1+1)-ES loop with periodic restarts
        sigma = 2.0
        no_improve = 0
        max_no_improve = 40
        restart_count = 0
        max_restarts = 5

        while calls < budget:
            # Time check
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < best_f:
                best_x, best_f = y, fy
                no_improve = 0
                # Increase step size on success
                sigma *= 1.2
            else:
                no_improve += 1
                # Decrease step size on failure
                sigma *= 0.9

            # Keep sigma in reasonable bounds
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # Restart if stuck
            if no_improve >= max_no_improve and restart_count < max_restarts:
                restart_count += 1
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.5)
                # Try a new random point, but keep the best found so far
                x_new = rng.uniform(-5, 5, dim)
                fx_new = func(x_new)
                calls += 1
                if fx_new < best_f:
                    best_x, best_f = x_new, fx_new
                else:
                    # Stay near the best point but with a larger step
                    best_x = best_x + 2.0 * rng.standard_normal(dim)
                    best_x = np.clip(best_x, -5, 5)
                sigma = max(2.0, sigma)

            # If we've used up restarts and still stuck, just continue
            # The step size will keep shrinking, which is fine

        return best_x, best_f
# EVOLVE-BLOCK-END
