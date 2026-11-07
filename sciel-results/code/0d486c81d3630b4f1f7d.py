"""Constrained random search within shrinking hypersphere for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Constrained random search: sample within a shrinking hypersphere centered at origin."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # Use global numpy random state to respect evaluator's seeding
        # Adjust radii: Start smaller to focus more on the region near origin
        r_init = 10.0
        r_final = 0.1

        # Track non-dominated points
        F_list = []  # list of objective vectors
        X_list = []  # list of decision vectors

        for i in range(budget):
            # Linearly shrink radius
            t = i / max(1, budget - 1)
            r = r_init + (r_final - r_init) * t

            # Sample a point uniformly within the hypersphere of radius r centered at origin
            direction = np.random.randn(dim)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                direction = np.ones(dim) / np.sqrt(dim)
            else:
                direction /= norm
            u = np.random.rand()
            radius = r * (u ** (1.0 / dim))
            x = direction * radius

            # Clip to bounds
            x = np.clip(x, lo, hi)

            # Evaluate
            f = np.array(problem(x), dtype=float)

            # Store
            F_list.append(f)
            X_list.append(x)

        return X_list
# EVOLVE-BLOCK-END
