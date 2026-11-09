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

        # Initial step size - larger to explore the [-5,5]^5 space more aggressively
        sigma = 1.5

        # Adaptation factors for 1/5 success rule
        c_inc = 1.15
        c_dec = 1.0 / 1.15

        # Phase 1: main search
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

            # If sigma is very small, we're close to convergence - stop early
            if sigma < 1e-6 and calls > 100:
                break

        # Phase 2: polish - use a very small step size to refine
        remaining = self.budget - calls
        if remaining > 10 and sigma < 0.1:
            sigma_polish = max(sigma * 0.1, 1e-7)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish = np.clip(sigma_polish * 1.1, 1e-9, 1.0)
                else:
                    sigma_polish = np.clip(sigma_polish / 1.1, 1e-9, 1.0)

        return x, fx
# EVOLVE-BLOCK-END
