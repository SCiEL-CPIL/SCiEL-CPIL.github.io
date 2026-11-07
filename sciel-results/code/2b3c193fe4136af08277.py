"""Optimized optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pure (1+1)-ES with 1.5/0.87 adaptation rates, initialized at center."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Initialize at the box center for maximum early convergence speed
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.0

        # Pure (1+1)-ES with 1.5/0.87 adaptation rates
        for _ in range(budget - 1):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
            else:
                sigma *= 0.87

            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
