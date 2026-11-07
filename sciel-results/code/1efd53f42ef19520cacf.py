"""Optimized optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for f13 Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start initialization
        # Evaluate 50 random points to find promising basins
        n_init = 50
        init_points = []
        init_fvals = []

        for i in range(n_init):
            if time.time() > deadline:
                break
            if budget - len(init_points) <= 0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            init_points.append(x)
            init_fvals.append(fx)

        # Sort by fitness and pick top 5
        order = np.argsort(init_fvals)
        n_starts = min(5, len(order))
        start_indices = order[:n_starts]

        # Phase 2: Run (1+1)-ES from each of the top 5 starting points
        # Allocate remaining budget evenly
        remaining_budget = budget - len(init_points)
        budget_per_start = max(1, remaining_budget // n_starts)

        global_best_x = init_points[start_indices[0]].copy()
        global_best_fx = init_fvals[start_indices[0]]

        for si in range(n_starts):
            if time.time() > deadline:
                break

            x = init_points[start_indices[si]].copy()
            fx = init_fvals[start_indices[si]]
            sigma = 1.0

            # (1+1)-ES with 1/5 success rule
            for _ in range(budget_per_start):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87

                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 10.0)

                if fx < global_best_fx:
                    global_best_fx = fx
                    global_best_x = x.copy()

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
