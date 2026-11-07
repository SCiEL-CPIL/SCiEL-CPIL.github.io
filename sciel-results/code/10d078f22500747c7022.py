"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_fx = fx
        best_x = x.copy()

        sigma = 2.0
        # 1/5 rule parameters
        window = []

        calls = 1
        while calls < self.budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                window.append(1)
                x, fx = y, fy
            else:
                window.append(0)

            # Keep only last 10 steps for 1/5 rule
            if len(window) > 10:
                window.pop(0)

            # Update sigma based on 1/5 success rule
            if len(window) >= 10:
                success_rate = sum(window[-10:]) / 10.0
                if success_rate > 0.3:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.75

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
