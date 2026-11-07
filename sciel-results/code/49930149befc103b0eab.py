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
                # Increase step size on success (1/5 rule: multiply by ~1.35)
                sigma *= 1.35
            else:
                no_improve += 1
                # Decrease step size on failure
                sigma *= 0.75

            # Keep sigma in reasonable bounds
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 15.0)

            # Restart if stuck
            if no_improve >= max_no_improve and restart_count < max_restarts:
                restart_count += 1
                no_improve = 0
                # Increase the patience for next restart
                max_no_improve = int(max_no_improve * 1.4)
                
                # Strategy: try a new random point far from current best
                x_new = rng.uniform(-5, 5, dim)
                fx_new = func(x_new)
                calls += 1
                
                if fx_new < best_f:
                    best_x, best_f = x_new, fx_new
                    sigma = 3.0
                else:
                    # Stay near the best point but with a larger step for exploration
                    # Jump to a point that's somewhat far but not completely random
                    jump_scale = 2.0 + restart_count * 0.5
                    best_x = np.clip(best_x + jump_scale * rng.standard_normal(dim), -5, 5)
                    sigma = max(3.0, sigma)

            # If we've used up restarts, just continue with shrinking sigma
            # This allows fine-tuning around the best found point

        # Final polish phase: coordinate-wise local search with fine steps
        if calls < budget and time.time() < deadline:
            polish_calls = min(budget - calls, 100)
            # Try coordinate-wise perturbations with decreasing step sizes
            for step in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
                for d in range(dim):
                    for direction in [-1, 1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = best_x.copy()
                        y[d] = np.clip(y[d] + direction * step, -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < best_f:
                            best_x, best_f = y, fy

        return best_x, best_f
# EVOLVE-BLOCK-END
