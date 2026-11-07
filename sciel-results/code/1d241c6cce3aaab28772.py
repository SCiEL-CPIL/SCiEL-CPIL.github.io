"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s kill

        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        sigma = 2.0
        calls = 1

        # Main (1+1)-ES loop
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
                if sigma < 1e-8:
                    break

        # Final polish: refine around best point with smaller sigma
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            sigma = max(sigma, 0.1)
            for _ in range(min(remaining - 5, 50)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    if sigma < 1e-9:
                        break

        return x, fx
# EVOLVE-BLOCK-END
