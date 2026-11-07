"""New program using a two-phase optimizer: global random search followed by adaptive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Two-phase optimizer: global exploration then local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(123)
        best_x = None
        best_f = np.inf
        calls = 0
        budget = self.budget
        dim = self.dim

        # Phase 1: Global exploration with random sampling (15% of budget)
        explore_budget = max(20, int(budget * 0.15))
        for i in range(explore_budget):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # Phase 2: Local refinement with adaptive (1+1)-ES
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f

        x = best_x.copy()
        fx = best_f
        sigma = 1.5  # Start with a moderate step size

        for i in range(remaining_budget):
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            else:
                sigma *= 0.92

            # Periodic escape attempts from local minima
            if i % 30 == 29:
                y_jump = np.clip(x + 1.5 * rng.standard_normal(dim), -5, 5)
                fy_jump = func(y_jump)
                calls += 1
                if fy_jump < fx:
                    x, fx = y_jump, fy_jump
                    sigma = 1.0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
