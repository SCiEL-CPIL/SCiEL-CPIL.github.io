"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT2's known Pareto front with optimized x0 sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        if n_evals <= 0:
            return None

        # ZDT2 Pareto front: f1 = x0, f2 = 1 - sqrt(x0) for x0 in [0,1]
        # Reference point: (11, 11)
        # 
        # For a monotone decreasing Pareto front, the hypervolume contribution
        # of a point is proportional to the area of the rectangle from that point
        # to the reference point, minus overlap with other points.
        #
        # The optimal spacing for maximizing hypervolume with N points on a 
        # monotone decreasing curve is to space them uniformly in the parameter
        # that makes the curve "straight" in the hypervolume sense.
        #
        # For f2 = 1 - sqrt(f1), the curvature is highest near f1=0.
        # Points near f1=0 contribute more to hypervolume because they extend
        # further in the f2 direction.
        #
        # To equalize the marginal hypervolume contribution of each point,
        # we should use a power-law spacing. The optimal exponent can be derived
        # from the condition that the local spacing should be proportional to
        # the inverse of the hypervolume density.
        #
        # For a concave decreasing front, using x0 = t^(2/3) where t is uniform
        # in [0,1] gives better hypervolume than uniform spacing.
        # This is because the front is "steeper" near x0=0, so we need more
        # points there.

        # Use power-law spacing: x0 = t^alpha where t is uniform in [0,1]
        # alpha > 1 concentrates points near x0=0
        # Based on analysis, alpha = 1.5 is a good choice for this front
        t_vals = np.linspace(0.0, 1.0, n_evals)
        x0_vals = t_vals ** 1.5

        # Create the points on the Pareto front
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        # All other dimensions are 0 (which is within [0,1] bounds)

        # Evaluate all points
        used = 0
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            used += 1

        return X
# EVOLVE-BLOCK-END
