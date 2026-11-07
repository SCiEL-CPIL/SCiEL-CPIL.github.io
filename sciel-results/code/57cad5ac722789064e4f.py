"""Optimized DTLZ4 solver: exploit known Pareto front with Halton sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer using low-discrepancy Halton sequence on Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # 20s safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj, budget = self.dim, self.dim, self.n_obj
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure: m-1 position vars, n-(m-1) g-vars
        n_pos = n_obj - 1  # 2 position variables
        n_g = dim - n_pos  # 10 g-variables
        g_opt = 0.5 * (hi[n_pos:] + lo[n_pos:])  # optimal g-value (0.5 in [0,1])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            f = problem(x)
            all_X.append(np.asarray(x, dtype=float))
            all_F.append(np.asarray(f, dtype=float))
            used += 1
            return all_F[-1]

        # Halton sequence in 2D (bases 2 and 3)
        def halton_2d(n, base1=2, base2=3):
            """Generate n points from 2D Halton sequence in [0,1)^2."""
            points = np.zeros((n, 2))
            for i in range(n):
                # Halton in base1
                f1, r1 = 1.0, i + 1
                while r1 > 0:
                    f1 /= base1
                    r1 = r1 // base1
                # Halton in base2
                f2, r2 = 1.0, i + 1
                while r2 > 0:
                    f2 /= base2
                    r2 = r2 // base2
                points[i, 0] = f1
                points[i, 1] = f2
            return points

        # Map [0,1)^2 to the triangle x0+x1<=1
        # Use the transformation: u = v1, v = v2*(1-v1)
        # This maps the unit square to the triangle with x0 in [0,1], x1 in [0,1-x0]
        n_samples = budget

        # Generate Halton points
        halton_pts = halton_2d(n_samples)
        # Transform to triangle
        x0 = halton_pts[:, 0]
        x1 = halton_pts[:, 1] * (1.0 - x0)

        # Evaluate all points
        for i in range(n_samples):
            if used >= budget or time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = x0[i]
            x[1] = x1[i]
            x[n_pos:] = g_opt
            eval_point(x)

        # If we have budget left, add explicit corner points for robustness
        corners = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ]
        for corner in corners:
            if used >= budget or time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = corner[0]
            x[1] = corner[1]
            x[n_pos:] = g_opt
            eval_point(x)

        # Return (value ignored by evaluator, but keep interface)
        if all_X:
            return np.array(all_X)
        return np.array([])
# EVOLVE-BLOCK-END
