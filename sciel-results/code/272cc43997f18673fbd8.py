"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Lattice global search followed by adaptive (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # Phase 1: Deterministic lattice search on a 3^5 grid
        # This gives 243 evaluations covering the entire [-5, 5]^5 space
        levels = np.linspace(-5.0, 5.0, 3)  # [-5, 0, 5]
        grid_points = np.array(list(itertools.product(*[levels] * n)))

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Evaluate all grid points
        for i in range(len(grid_points)):
            if time.time() > deadline:
                break
            x = grid_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement using adaptive (1+1)-ES from best grid point
        remaining_budget = budget - calls_made
        if remaining_budget > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            rng = np.random.default_rng(42)

            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if sigma > 5.0:
                        sigma = 5.0
                else:
                    sigma *= 0.85
                    if sigma < 0.01:
                        sigma = 0.01

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
