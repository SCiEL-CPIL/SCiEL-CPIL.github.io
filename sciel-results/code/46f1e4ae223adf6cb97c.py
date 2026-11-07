"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """(1+1)-ES with adaptive step size and time guard for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Initial point: center of the box (good for BBOB functions)
        x = np.zeros(dim)
        fx = func(x)
        calls_made = 1

        # Initial step size
        sigma = 1.0
        best_x = x.copy()
        best_f = fx

        # Success history for 1/5 rule
        successes = 0
        window = 10
        sigma_history = []

        for i in range(budget - 1):
            if time.time() >= deadline:
                break

            # Generate perturbation
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1

            # Track best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Accept/reject
            if fy < fx:
                x = y
                fx = fy
                successes += 1
                sigma *= 1.2  # increase step size on success
            else:
                sigma *= 0.8  # decrease step size on failure

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Periodic restart if stuck (no improvement in last 50 calls)
            if i > 0 and i % 50 == 0:
                # Check if we've made recent progress
                if fx > best_f + 1e-10:
                    # We're stuck at a local optimum, try a small random jump
                    jump = 0.5 * rng.standard_normal(dim)
                    y = np.clip(x + jump, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
