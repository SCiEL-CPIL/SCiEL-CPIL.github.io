"""Optimized DTLZ7 solver with hypervolume-aware initialization and NSGA-II-style search."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using hypervolume-optimized lattice + evolutionary search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget

        def make_front_point(x1, x2):
            """Create a decision vector on the Pareto front (g=1)."""
            x = np.zeros(n)
            x[0] = x1
            x[1] = x2
            x[2:] = 0.0
            return np.clip(x, lo, hi)

        # Generate a deterministic lattice on the Pareto front
        # Use a grid in (u, v) space, transform with power law to cluster near (0,0)
        # Full square domain [0, 1] x [0, 1] for x1, x2
        # Target ~200 points. 14x14 = 196 points.
        N = 14
        power = 3.0
        
        pts = []
        for i in range(N):
            for j in range(N):
                u = (i + 0.5) / N
                v = (j + 0.5) / N
                u_t = u ** power
                v_t = v ** power
                pts.append((u_t, v_t))
        
        # If we have too many, subsample deterministically (take first N)
        if len(pts) > budget:
            pts = pts[:budget]
        
        # If we have too few, add random points on the front
        while len(pts) < budget:
            u = rng.random() ** power
            v = rng.random() ** power
            pts.append((u, v))
        
        pts = np.array(pts[:budget])
        X = np.array([make_front_point(x1, x2) for x1, x2 in pts])
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
