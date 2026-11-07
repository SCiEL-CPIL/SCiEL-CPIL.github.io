"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with windowed 1/5 success rule and time guard for BBOB f3."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # Start at origin — BBOB shifts are typically small, so this is often close
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 1.0
        successes = 0
        attempts = 0
        window_size = 10

        for i in range(budget - 1):
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                if fy < best_f:
                    best_f = fy
                    best_x = x.copy()
            else:
                pass

            attempts += 1

            # Adapt sigma every window_size steps using 1/5 success rule
            if attempts % window_size == 0:
                success_rate = successes / attempts
                if success_rate > 0.2:
                    sigma *= 1.4
                elif success_rate < 0.2:
                    sigma *= 0.7
                sigma = np.clip(sigma, 0.05, 5.0)
                successes = 0
                attempts = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
