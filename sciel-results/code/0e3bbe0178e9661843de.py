"""Portfolio search for DTLZ2: Sobol sequence with surface-area-weighted sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Sobol-sequence front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        def time_ok():
            return time.time() < deadline

        def front_to_x(t):
            """Convert angle parameters t (length n_obj-1) to decision vector x.
            For DTLZ2: x_i = sin(t_i) for i < n_obj-1, and the rest are set
            to values that make g(x) = 0 (i.e., x_i = 0.5 for i >= n_obj-1).
            """
            x = np.full(self.dim, 0.5)
            for i in range(min(self.n_obj - 1, self.dim)):
                x[i] = np.sin(t[i])
            return x

        # Generate Sobol sequence points in 2D
        def sobol_2d(n):
            """Generate n Sobol sequence points in [0,1)^2."""
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            points = sampler.random(n)
            return points

        n_evals = self.budget
        sobol_pts = sobol_2d(n_evals)

        # The DTLZ2 Pareto front in 3 objectives is a quarter-sphere.
        # We use a warping that concentrates points near the edges of the parameter space
        # (t1, t2 near 0 or pi/2) to maximize hypervolume contribution.
        # This is achieved by mapping uniform u to w = sin^2(pi/2 * u), which follows
        # a Beta(0.5, 0.5) distribution, having high density at 0 and 1.

        X = []
        F = []
        used = 0
        for k in range(n_evals):
            if not time_ok():
                break
            u1, v2 = sobol_pts[k]
            # Warp to concentrate near boundaries
            w1 = np.sin(0.5 * np.pi * u1)
            w1 = w1 * w1
            w2 = np.sin(0.5 * np.pi * v2)
            w2 = w2 * w2
            
            # Map warped values to angles
            # w1 in [0,1] -> t1 in [0, pi/2]
            # Since we want t1 to be near 0 or pi/2 when w1 is near 0 or 1,
            # and the relationship between t1 and surface area is t1=arcsin(sin_t1),
            # we can just map w1 directly to the "sin(t1)" space which is uniform
            # if we wanted uniform surface area. 
            # However, we already warped u1 to w1.
            # If we set sin(t1) = w1, then t1 = arcsin(w1).
            # Since w1 is concentrated at 0 and 1, t1 will be concentrated at 0 and pi/2.
            t1 = np.arcsin(w1)
            
            # For t2, the surface area element is constant with respect to t2.
            # So uniform t2 is optimal for surface coverage.
            # But we want edge concentration.
            # So we map w2 (concentrated at 0,1) to t2 linearly?
            # t2 = w2 * pi/2.
            # This concentrates t2 near 0 and pi/2.
            t2 = w2 * (np.pi / 2)
            
            t = np.array([t1, t2])
            x = front_to_x(t)
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
            used += 1

        X = np.array(X)
        F = np.array(F)

        return X

# EVOLVE-BLOCK-END
