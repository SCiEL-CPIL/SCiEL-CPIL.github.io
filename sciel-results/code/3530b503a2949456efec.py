"""ZDT2 Optimizer using geometric decomposition: curvature-adaptive grid spacing on the analytic Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Geometric decomposition of the objective space.
    1. Identify the 1D Pareto front of ZDT2: x[1:]=0, x[0] in [0,1].
       f1 = x[0]
       f2 = (1 + 10*(dim-1)) * (1 - sqrt(f1 / (1 + 10*(dim-1))))
    2. The hypervolume contribution of a point (f1_i, f2_i) is determined by the rectangle
       it encloses with the reference point and its neighbors.
    3. For a concave front, the optimal spacing is non-uniform: points are denser where
       the curvature is high (i.e., where |f2''| is large).
    4. We use the arc-length parameterization of the front to derive a curvature-adaptive
       grid: sample points such that the change in tangent angle is approximately uniform,
       which naturally places more points in high-curvature regions.
    5. This is done analytically without iterative numerical inversion.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        dim = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT2 specifics
        n_extra = dim - 1
        scale = 1.0 + 10.0 * n_extra

        # Pareto front: x[0] = t, x[1:] = 0
        # f1(t) = t
        # f2(t) = scale * (1 - sqrt(t / scale))
        
        # Reference point
        ref_f1 = 11.0
        ref_f2 = 11.0

        N = budget

        # Curvature-adaptive sampling:
        # The front is f2(t) = scale - scale * sqrt(t/scale) = scale - sqrt(scale * t)
        # Let's parameterize by the tangent angle or use a transformation that
        # equalizes the curvature.
        #
        # A simpler and effective approach: sample in the space of sqrt(t).
        # Let u = sqrt(t), so t = u^2, u in [0, 1].
        # f1 = u^2
        # f2 = scale * (1 - u / sqrt(scale)) = scale - sqrt(scale) * u
        # This is a LINE in (f1, f2) space! 
        # Wait, f1 = u^2 and f2 = scale - sqrt(scale)*u.
        # So f2 = scale - sqrt(scale) * sqrt(f1).
        # This is still a curve in (f1, f2).
        #
        # Actually, if we sample u uniformly in [0, 1], we get:
        # f1 = u^2 (quadratic spacing in f1)
        # f2 = linear in u
        # This places more points where f1 is small (high curvature region).
        # This is exactly the curvature-adaptive spacing we want.
        
        # Let's verify: the curvature of f2 vs f1:
        # f2' = -0.5 * sqrt(scale / f1)
        # f2'' = 0.25 * scale / (f1 * sqrt(scale)) = 0.25 * sqrt(scale) / f1
        # Curvature is high when f1 is small.
        # Uniform u gives f1 = u^2, so df1 = 2u du.
        # For small f1 (small u), df1 is small, so points are dense. Correct.
        
        if N <= 1:
            u_vals = np.array([0.0])
        else:
            # Sample u uniformly in [0, 1]
            u_vals = np.linspace(0.0, 1.0, N)
        
        # Compute f1 values: t = u^2
        t_vals = u_vals ** 2
        
        # Ensure endpoints are exactly 0 and 1
        t_vals[0] = 0.0
        t_vals[-1] = 1.0

        # Evaluate the points
        last_F = None
        for t in t_vals:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = np.clip(t, lo[0], hi[0])
            # x[1:] = 0 (clipped to bounds, which are [0,1])
            F = np.array(problem(x))
            last_F = F

        return last_F
# EVOLVE-BLOCK-END
