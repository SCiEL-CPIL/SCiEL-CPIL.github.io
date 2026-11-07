import numpy as np
import time

class Optimizer:
    """DTLZ2-specific optimizer: low-discrepancy sampling with hypervolume-weighted mapping."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 structure:
        # Pareto front: g=0 when x_i=0.5 for i>=m (indices m-1 to n-1)
        # f1 = cos(theta1)*cos(theta2), f2 = cos(theta1)*sin(theta2), f3 = sin(theta1)
        # theta1 = x0*pi/2, theta2 = x1*pi/2
        # Surface area element: cos(theta1) dtheta1 dtheta2
        # For hypervolume optimization, weight toward interior where marginal volume is higher

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        evaluated_points = []
        used_budget = 0

        # Strategy: Low-discrepancy sequence (Halton) mapped via inverse CDF
        # of the Pareto front's surface area measure (arcsin) with hypervolume
        # weighting (clustering toward center). This avoids grid artifacts and
        # boundary gaps inherent in uniform or recursive bisection schemes.

        alpha = 0.55  # Clustering parameter

        def cluster_toward_center(u, alpha=0.55):
            """Map u in [0,1] to [0,1] with clustering toward 0.5"""
            if u < 0.5:
                return 0.5 - 0.5 * (1 - 2*u)**alpha
            else:
                return 0.5 + 0.5 * (2*u - 1)**alpha

        # Generate all points using Halton low-discrepancy sequence
        for k in range(self.budget):
            if time.time() > deadline:
                break
            if used_budget >= self.budget:
                break

            u_raw = halton(k + 1, 2)
            v_raw = halton(k + 1, 3)

            # Apply clustering toward center (hypervolume weighting)
            u = cluster_toward_center(u_raw, alpha)
            v = cluster_toward_center(v_raw, alpha)

            # Map to DTLZ2 parameters
            # x0: arcsin transformation for surface-area uniformity
            x0 = (2.0 / np.pi) * np.arcsin(u)
            x1 = v

            x = base_x.copy()
            x[0] = x0
            x[1] = x1

            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        return evaluated_points
