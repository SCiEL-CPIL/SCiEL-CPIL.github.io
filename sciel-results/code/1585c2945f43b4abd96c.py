"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with time-aware budget management and stagnation restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 330s kill - 40s safety margin

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        calls_used = 1
        best_x = x.copy()
        best_f = fx
        stagnation_counter = 0
        no_improve_threshold = 50

        # Main (1+1)-ES loop with time guard
        while calls_used < self.budget:
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                stagnation_counter = 0

                # Track global best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                sigma *= 0.85
                stagnation_counter += 1

            # If sigma becomes too small, or stagnation detected, restart exploration
            if sigma < 1e-6 or stagnation_counter > no_improve_threshold:
                # Try random restarts if we still have budget
                restarts = min(15, self.budget - calls_used)
                for _ in range(restarts):
                    if time.time() >= deadline:
                        break
                    y = rng.uniform(-5, 5, self.dim)
                    fy = func(y)
                    calls_used += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        # If we found a better point, restart from there
                        x = y.copy()
                        fx = fy
                        sigma = 2.0
                        stagnation_counter = 0
                        break
                else:
                    # No improvement found, reset sigma and continue
                    sigma = 2.0
                    stagnation_counter = 0

                if time.time() >= deadline:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
