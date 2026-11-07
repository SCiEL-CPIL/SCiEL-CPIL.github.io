"""Optimizer for BBOB-style Schwefel f20 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, coordinate descent, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.5
        calls = 1

        # Success tracking for adaptive sigma
        successes = 0
        window = 10

        for i in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                sigma *= 1.2
            else:
                sigma *= 0.9

            # Keep sigma in reasonable bounds
            sigma = max(sigma, 1e-4)
            sigma = min(sigma, 5.0)

            # Periodic coordinate descent every 50 calls
            if (i + 1) % 50 == 0:
                if time.time() > deadline:
                    break
                # Coordinate descent on current best
                cd_x = x.copy()
                cd_fx = fx
                for coord in range(dim):
                    if time.time() > deadline:
                        break
                    # Try small steps along this coordinate
                    for step_size in [0.5, 0.2, 0.1, 0.05]:
                        for direction in [1, -1]:
                            trial = cd_x.copy()
                            trial[coord] += direction * step_size
                            trial = np.clip(trial, -5, 5)
                            ft = func(trial)
                            calls += 1
                            if ft < cd_fx:
                                cd_x = trial
                                cd_fx = ft
                    if time.time() > deadline:
                        break
                if cd_fx < fx:
                    x = cd_x
                    fx = cd_fx
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
