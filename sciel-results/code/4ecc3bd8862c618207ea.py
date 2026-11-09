"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 320.0  # safety margin before 330s kill

        # Start at the center of the box
        x = np.zeros(self.dim)
        fx = func(x)
        calls = 1

        # Initial step size
        sigma = 1.0

        # Adaptation factors for 1/5 success rule (gentler for stability)
        c_inc = 1.1
        c_dec = 1.0 / 1.1

        for _ in range(self.budget - 1):
            # Deadline guard
            if time.time() > deadline:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= c_inc
            else:
                sigma *= c_dec

            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
