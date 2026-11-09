"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time-aware."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0

        calls_made = 1
        best_x = x.copy()
        best_f = fx

        for i in range(self.budget - 1):
            # Check time budget
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.5
            else:
                sigma *= 0.87

        return best_x, best_f
# EVOLVE-BLOCK-END
