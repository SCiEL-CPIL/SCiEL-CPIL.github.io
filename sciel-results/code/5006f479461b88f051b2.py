"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-resolution grid descent with adaptive step sizes for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget
        calls_made = 0

        # Initial point
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        calls_made += 1
        best_x = x.copy()
        best_fx = fx

        # Multi-resolution grid descent
        # We'll try different step sizes, starting large and getting smaller
        # For each step size, we try moving in each coordinate direction

        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                      0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125,
                      0.000244140625, 0.0001220703125, 0.00006103515625, 0.000030517578125,
                      0.0000152587890625, 0.00000762939453125, 0.000003814697265625,
                      0.0000019073486328125, 0.00000095367431640625]

        for step in step_sizes:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            improved = True
            while improved and calls_made < budget:
                if time.time() > deadline:
                    break
                improved = False
                for d in range(n):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break

                    # Try positive direction
                    y = x.copy()
                    y[d] = min(5.0, x[d] + step)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        if fx < best_fx:
                            best_x, best_fx = x.copy(), fx
                        continue

                    # Try negative direction
                    y = x.copy()
                    y[d] = max(-5.0, x[d] - step)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        if fx < best_fx:
                            best_x, best_fx = x.copy(), fx

        # Use remaining budget for fine-tuning with random perturbations
        if calls_made < budget:
            sigma = 0.001
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
