"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with 1/5 success rule for f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Initialize at origin (often near the attractor for f6)
        x = np.zeros(dim)
        fx = func(x)
        calls_made = 1
        best_x = x.copy()
        best_f = fx

        # Initial sigma: large enough to explore the [-5,5]^5 space
        sigma = 4.0

        # 1/5 success rule parameters
        successes = 0
        total = 0

        while calls_made < budget:
            if time.time() >= deadline:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            else:
                pass  # keep x

            total += 1

            # Update sigma every 10 steps using 1/5 success rule
            if total % 10 == 0:
                success_rate = successes / total
                if success_rate > 0.2:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.87
                successes = 0
                total = 0

            # Stop if sigma is too small
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
