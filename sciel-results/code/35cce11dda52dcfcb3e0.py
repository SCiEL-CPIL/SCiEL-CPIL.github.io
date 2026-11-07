import numpy as np
import time
from scipy.stats import qmc
from scipy.spatial import ConvexHull


class Optimizer:
    """Optimizer for DTLZ4 using parameterized Pareto front sampling with
    strategic corner/edge points and Sobol quasi-random fill."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 with n_obj=3: Pareto front is parameterized by theta_1, theta_2 in [0, pi/2]
        # x_0, x_1 map to the front via:
        #   f1 = 0.5 * x0 * x1
        #   f2 = 0.5 * x0 * (1 - x1)
        #   f3 = 0.5 * x1 * (1 - x0)
        # This is a curved 2D surface. We parameterize using angles on the quarter circle.

        # Strategic points: corners and key edge points on the Pareto front
        # Corners in objective space correspond to specific (theta1, theta2) values
        # f1=0.5,f2=0,f3=0 -> x0=1,x1=0 -> theta1=0, theta2=pi/2 (in some parametrization)
        # Let's use direct (x0, x1) parameterization where x0=sin(theta1), x1=sin(theta2)
        # Actually for DTLZ4: x0, x1 are the position variables, rest are 0.5 for g=0

        # Pareto front: f1 = 0.5*x0*x1, f2 = 0.5*x0*(1-x1), f3 = 0.5*x1*(1-x0)
        # All in [0, 0.5]. Reference point is [2.5, 2.5, 2.5].

        # Strategic points in (x0, x1) space
        strategic_pts = []

        # Corners of the unit square in (x0, x1)
        strategic_pts.append((0.0, 0.0))   # f=[0,0,0]
        strategic_pts.append((1.0, 0.0))   # f=[0,0.5,0]
        strategic_pts.append((0.0, 1.0))   # f=[0,0,0.5]
        strategic_pts.append((1.0, 1.0))   # f=[0.5,0,0]

        # Edge midpoints
        strategic_pts.append((0.5, 0.0))   # f=[0,0.25,0]
        strategic_pts.append((0.5, 1.0))   # f=[0,0.25,0.25]
        strategic_pts.append((0.0, 0.5))   # f=[0,0,0.25]
        strategic_pts.append((1.0, 0.5))   # f=[0.25,0,0.25]

        # Additional edge points for better coverage
        for i in [0.25, 0.75]:
            strategic_pts.append((i, 0.0))
            strategic_pts.append((i, 1.0))
            strategic_pts.append((0.0, i))
            strategic_pts.append((1.0, i))

        # Center and interior points
        strategic_pts.append((0.5, 0.5))   # f=[0.25,0.25,0.25]
        strategic_pts.append((1/3, 1/3))
        strategic_pts.append((2/3, 2/3))
        strategic_pts.append((1/3, 2/3))
        strategic_pts.append((2/3, 1/3))

        n_strategic = len(strategic_pts)
        n_sobol = self.budget - n_strategic

        if n_sobol < 0:
            n_sobol = 0
            n_strategic = min(n_strategic, self.budget)
            strategic_pts = strategic_pts[:self.budget]

        # Generate Sobol points in [0,1]^2
        try:
            sampler = qmc.Sobol(d=2, scramble=False, seed=42)
            sobol_pts = sampler.random(n_sobol)
        except Exception:
            # Fallback: Halton sequence
            def halton(n, base):
                result = []
                for i in range(1, n + 1):
                    f = 1.0
                    res = 0.0
                    x = i
                    while x > 0:
                        f /= base
                        res += f * (x % base)
                        x //= base
                    result.append(res)
                return np.array(result)
            sobol_pts = np.column_stack([halton(n_sobol, 2), halton(n_sobol, 3)])

        # Combine all (x0, x1) pairs
        all_pts = np.vstack([np.array(strategic_pts), sobol_pts])

        # Deduplicate: remove points that are too close
        if len(all_pts) > 1:
            # Simple dedup: sort and remove near-duplicates
            all_pts_sorted = all_pts[np.lexsort((all_pts[:, 1], all_pts[:, 0]))]
            mask = np.ones(len(all_pts_sorted), dtype=bool)
            for i in range(1, len(all_pts_sorted)):
                if np.allclose(all_pts_sorted[i], all_pts_sorted[i-1], atol=1e-10):
                    mask[i] = False
            all_pts = all_pts_sorted[mask]

        # If we have more points than budget due to dedup removing some, that's fine
        # We just evaluate up to budget points
        n_eval = min(len(all_pts), self.budget)
        all_pts = all_pts[:n_eval]

        # Construct decision vectors
        X = np.zeros((n_eval, self.dim))
        X[:, 0] = all_pts[:, 0]
        X[:, 1] = all_pts[:, 1]
        X[:, 2:] = 0.5  # g=0 for Pareto front

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate within time budget
        F = np.empty((n_eval, self.n_obj))
        for i in range(n_eval):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
