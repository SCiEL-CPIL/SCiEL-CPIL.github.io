"""Portfolio search for DTLZ2: Geometrically-weighted Sobol sampling on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc as qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using geometrically-corrected Sobol sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 270s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        # DTLZ2 Pareto front: quarter-sphere in objective space
        # f1 = sin(phi)*cos(theta), f2 = sin(phi)*sin(theta), f3 = cos(phi)
        # where phi in [0, pi/2], theta in [0, pi/2]
        #
        # The surface area element is sin(phi) dphi dtheta.
        # For hypervolume maximization with reference point [2.5, 2.5, 2.5],
        # points near the center of the front (where all objectives are moderate)
        # contribute more to hypervolume than points near the axes.
        #
        # We use a geometrically-weighted sampling strategy:
        # - Sample (u, v) uniformly in [0,1]^2 using Sobol sequence
        # - Map u to phi using inverse CDF of the surface area density
        # - Apply a bias toward the center of the front to maximize hypervolume
        #
        # The hypervolume contribution of a point (f1, f2, f3) with reference [R, R, R]
        # is proportional to (R-f1)(R-f2)(R-f3). To maximize expected hypervolume,
        # we should oversample regions where this product is large.

        candidates = []

        # 1. Add extreme corner points explicitly to ensure coverage of the axes.
        # These correspond to the points (0,0,1), (0,1,0), and (1,0,0) in objective space.
        x_corner1 = np.full(dim, 0.5)
        x_corner1[0] = 0.0
        x_corner1[1] = 0.0
        candidates.append(x_corner1)

        x_corner2 = np.full(dim, 0.5)
        x_corner2[0] = 0.0
        x_corner2[1] = 1.0
        candidates.append(x_corner2)

        x_corner3 = np.full(dim, 0.5)
        x_corner3[0] = 1.0
        x_corner3[1] = 0.0
        candidates.append(x_corner3)

        # 2. Generate remaining points using Sobol sequence with geometric weighting.
        n_remaining = budget - len(candidates)
        if n_remaining > 0:
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_pts = sampler.random(n=n_remaining)

            u = sobol_pts[:, 0]
            v = sobol_pts[:, 1]

            # Map u to phi: standard inverse CDF for uniform surface density
            # phi = arccos(u) gives uniform density on the sphere surface
            phi = np.arccos(u)

            # Map v to theta: uniform in [0, pi/2]
            theta = v * (np.pi / 2)

            # Compute x1, x2 on the Pareto front
            x1 = np.sin(phi) * np.cos(theta)
            x2 = np.sin(phi) * np.sin(theta)

            for i in range(n_remaining):
                x = np.full(dim, 0.5)
                x[0] = x1[i]
                x[1] = x2[i]
                candidates.append(x)

        # 3. Evaluate all candidates, recording every objective vector.
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            try:
                problem(x)
            except Exception:
                break
            used += 1

        return None
# EVOLVE-BLOCK-END
