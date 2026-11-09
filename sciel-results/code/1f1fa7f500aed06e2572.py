"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    (1+1)-ES with 1/5 success rule for BBOB f14.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s limit

        rng = np.random.default_rng(42)
        calls_used = 0
        dim = self.dim
        budget = self.budget

        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        # Start at origin (likely near global minimum for f14)
        x = np.zeros(dim)
        fx = func(x)
        calls_used += 1

        # 1/5 success rule parameters
        sigma = 1.0
        success_window = 0
        window_size = 1 + int(np.floor(5 * dim / (1 + np.log(dim))))

        while not check_time():
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                success_window += 1
            else:
                success_window -= 1

            # 1/5 success rule: adjust sigma every window_size steps
            if success_window > 0:
                sigma *= (1 + 1.0 / window_size)
            elif success_window < 0:
                sigma *= (1 - 1.0 / window_size)
            success_window = 0

            # Clamp sigma to reasonable range
            sigma = max(1e-10, min(sigma, 10.0))

        return x, fx
# EVOLVE-BLOCK-END
