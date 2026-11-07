"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with zero-centered init, stable step-size adaptation, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        x = np.zeros(self.dim)
        fx = func(x)
        sigma = 2.5
        min_sigma = 1e-5
        max_sigma = 8.0
        no_improve = 0
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.15, max_sigma)
                no_improve = 0
            else:
                sigma = max(sigma * 0.92, min_sigma)
                no_improve += 1
                if no_improve >= 30:
                    # Stagnation: try a larger jump to escape potential local minima
                    sigma = min(sigma * 2.0, max_sigma)
                    no_improve = 0
        return x, fx
# EVOLVE-BLOCK-END
