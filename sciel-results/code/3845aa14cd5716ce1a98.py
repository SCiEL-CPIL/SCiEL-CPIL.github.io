"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # Initialize at center of box
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_used = 1

        sigma = 2.0
        no_improve = 0
        max_no_improve = 30

        for _ in range(budget - 1):
            if calls_used >= budget:
                break
            if time.time() > deadline - 5:
                break

            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                no_improve = 0
            else:
                sigma *= 0.9
                no_improve += 1

            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Restart if stuck
            if no_improve >= max_no_improve:
                x = np.clip(best_x + 0.5 * rng.standard_normal(n), -5, 5)
                fx = func(x)
                calls_used += 1
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                sigma = 2.0
                no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
