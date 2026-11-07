"""Constrained random search within shrinking hypersphere for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Constrained random search: sample within a shrinking hypersphere centered at origin."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # WFG9 Pareto front is near the origin in the decision space.
        # The bounds for WFG9 are typically [-50, 50] per dimension.
        # We sample within a hypersphere centered at the origin.
        # Radius shrinks linearly from an initial large value to a small value.

        # Initial radius: large enough to cover the space but focused near origin
        # For WFG9 with bounds [-50, 50], the Pareto front is very close to 0.
        # Let's use a radius that starts at ~20 and shrinks to ~0.1
        r_init = 20.0
        r_final = 0.05

        # Track non-dominated points
        F_list = []  # list of objective vectors
        X_list = []  # list of decision vectors

        for i in range(budget):
            # Linearly shrink radius
            t = i / max(1, budget - 1)
            r = r_init + (r_final - r_init) * t

            # Sample a point uniformly within the hypersphere of radius r centered at origin
            # Then clip to bounds
            # For uniform sampling in a hypersphere: sample direction uniformly on sphere, radius as r * u^(1/dim)
            # where u ~ Uniform(0,1)
            direction = rng.standard_normal(dim)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                direction = np.ones(dim) / np.sqrt(dim)
            else:
                direction /= norm
            u = rng.random()
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
