"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global multi-start (1+1)-ES for sharp ridge f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Phase 1: Global exploration - 50 random points
        n_initial = 50
        points = []
        for _ in range(n_initial):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            points.append((fx, x))

        # Keep top 5 starting points
        points.sort(key=lambda p: p[0])
        top_k = 5
        active = [(fx, x.copy()) for fx, x in points[:top_k]]

        best_val = active[0][0]
        best_x = active[0][1].copy()

        # Phase 2: Run (1+1)-ES from each of the 5 starting points
        # Allocate remaining budget evenly
        used = n_initial
        remaining = self.budget - used
        if remaining <= 0:
            return best_x, best_val

        per_es = remaining // top_k

        for idx in range(top_k):
            if time.time() > deadline:
                break

            fx, x = active[idx]
            sigma = 1.5

            # Adaptive sigma with 1/5 success rule rates
            sigma_up = 1.5
            sigma_down = 0.87

            for _ in range(per_es):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * sigma_up, 10.0)

                    if fx < best_val:
                        best_val = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * sigma_down, 0.01)

            # Update active list with improved point
            active[idx] = (fx, x.copy())

        return best_x, best_val
# EVOLVE-BLOCK-END
