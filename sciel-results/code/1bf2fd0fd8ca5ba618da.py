"""Rewritten optimizer using a CMA-ES-like adaptive strategy with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive evolution strategy with time guard for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0  # 300s limit, 20s safety margin

        rng = np.random.default_rng(42)

        # Initial point: center of the box (minimizer is in [-4,4]^5)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Initial step size: cover a large fraction of the box
        sigma = 4.0

        # Adaptive parameters
        step_increase = 1.2
        step_decrease = 0.8
        min_sigma = 1e-6
        max_sigma = 10.0

        # Track success for adaptive sigma
        successes = 0
        total_steps = 0

        call_count = 1
        calls_since_improve = 0

        for i in range(budget - 1):
            # Time guard
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                # Move to new point
                x = y
                fx = fy
                successes += 1
                calls_since_improve = 0

                # Update global best
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

                # Increase step size (with cap)
                sigma = min(sigma * step_increase, max_sigma)
            else:
                # Stay at current point
                calls_since_improve += 1

                # Decrease step size
                sigma = max(sigma * step_decrease, min_sigma)

            total_steps += 1

            # Adaptive restart: if stuck for too long, try a random restart
            if calls_since_improve > 50 and sigma < 0.1:
                # Random restart near best found
                x = best_x + 1.0 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                call_count += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                sigma = 2.0
                calls_since_improve = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
