"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with deadline guard for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s wall-clock limit, use 280s for safety
        dim = self.dim
        budget = self.budget

        # Phase 1: Global multi-start sampling
        # Evaluate 50 random points to identify promising basins
        n_initial = 50
        points = np.random.default_rng(42).uniform(-5, 5, (n_initial, dim))
        values = np.empty(n_initial)
        for i in range(n_initial):
            values[i] = func(points[i])

        # Keep top 5 starting points
        top_k = min(5, n_initial)
        top_indices = np.argsort(values)[:top_k]
        best_val = values[top_indices[0]]
        best_x = points[top_indices[0]].copy()

        # Phase 2: Run (1+1)-ES from each of the top 5 starting points
        # Allocate remaining budget evenly
        remaining = budget - n_initial
        per_es = max(1, remaining // top_k)

        for idx in top_indices:
            x = points[idx].copy()
            fx = values[idx]
            sigma = 1.5
            es_budget = per_es

            for _ in range(es_budget):
                # Deadline guard
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * np.random.default_rng(0).standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85

                # Update global best
                if fx < best_val:
                    best_val = fx
                    best_x = x.copy()

        return best_x, best_val
# EVOLVE-BLOCK-END
