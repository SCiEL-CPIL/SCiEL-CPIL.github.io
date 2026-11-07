"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        num_calls = 1
        sigma = 2.0

        def check_time():
            return time.time() > deadline

        for _ in range(budget - 1):
            if num_calls >= budget or check_time():
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            num_calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            sigma = max(sigma, 1e-6)

        return best_x, best_f
# EVOLVE-BLOCK-END
