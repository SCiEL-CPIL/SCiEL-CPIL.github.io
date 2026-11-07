"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 rule, multi-offspring escape, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s kill
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        sigma = 2.0
        calls = 1
        stuck_count = 0
        for _ in range(self.budget - 1):
            if time.time() > deadline or calls >= self.budget:
                break
            # If stuck for a while, try multi-offspring escape
            if stuck_count >= 30:
                # Try 3 offspring, pick the best
                best_y, best_fy = x.copy(), fx
                for _ in range(3):
                    if calls >= self.budget:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < best_fy:
                        best_y, best_fy = y, fy
                if best_fy < fx:
                    x, fx = best_y, best_fy
                    if fx < best_fx:
                        best_x, best_fx = x, fx
                    sigma *= 1.5
                    stuck_count = 0
                else:
                    sigma *= 0.85
                    stuck_count += 1
            else:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_x, best_fx = x, fx
                    sigma *= 1.5
                    stuck_count = 0
                else:
                    sigma *= 0.87
                    stuck_count += 1
        return best_x, best_fx
# EVOLVE-BLOCK-END
