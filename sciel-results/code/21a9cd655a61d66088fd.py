"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Phase 1: Global Exploration - sample 50 points to find the ridge
        n_explore = min(50, self.budget // 10)
        points = rng.uniform(-5, 5, (n_explore, self.dim))
        values = np.array([func(p) for p in points])

        # Check time after exploration
        if time.time() > deadline:
            best_idx = np.argmin(values)
            return points[best_idx], values[best_idx]

        # Select top 5 points as starting points for local exploitation
        top_k = min(5, n_explore)
        top_indices = np.argsort(values)[:top_k]
        top_points = points[top_indices].copy()
        top_values = values[top_indices].copy()

        best_x = top_points[0].copy()
        best_f = top_values[0]

        # Phase 2: Local Exploitation with parallel (1+1)-ES
        remaining_budget = self.budget - n_explore
        budget_per_es = max(1, remaining_budget // top_k)

        for i in range(top_k):
            if time.time() > deadline:
                break

            x = top_points[i].copy()
            fx = top_values[i]
            sigma = 2.0  # Start with larger step to navigate ridge structure

            # Run (1+1)-ES from this starting point
            for step in range(budget_per_es):
                if time.time() > deadline:
                    break

                # Generate offspring with adaptive step size
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3  # Moderate increase on success
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.75  # Decrease on failure

                # Early stopping if sigma becomes too small
                if sigma < 1e-4:
                    break

            # Update global best after each ES run
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
