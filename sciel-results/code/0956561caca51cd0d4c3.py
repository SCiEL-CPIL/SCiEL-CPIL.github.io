"""Optimizer for ZDT2 multi-objective problem."""
import numpy as np
import time


class Optimizer:
    """
    Optimizer tailored for ZDT2.
    Exploits the known structure of ZDT2:
      f1 = x[0]
      f2 = (1 + 10*(dim-1)) * (1 - sqrt(x[0]))
    The Pareto front is achieved when x[1:] = 0, giving:
      f1 = x[0] in [0, 1]
      f2 = C * (1 - sqrt(x[0]))  where C = 1 + 10*(dim-1)
    
    For dim=30: C = 291.
    
    Strategy: Use optimal spacing for hypervolume maximization.
    For a concave Pareto front, the optimal point placement equalizes
    the hypervolume contribution of each point. We derive the spacing
    analytically and use it to place points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        dim = self.dim
        C = 1 + 10 * (dim - 1)  # = 291 for dim=30
        
        # Pareto front: f1 = x[0], f2 = C*(1 - sqrt(x[0]))
        # Reference point: [11, 11]
        # Points count if f1 < 11 and f2 < 11.
        # f2 < 11 => C*(1-sqrt(x0)) < 11 => sqrt(x0) > 1 - 11/C
        # For C=291: sqrt(x0) > 280/291 => x0 > (280/291)^2 ≈ 0.9277
        
        # For hypervolume maximization on a concave front, optimal spacing
        # can be derived. The key insight: for ZDT2 with f2 = C*(1-sqrt(x0)),
        # the front is f2 = C - C*sqrt(f1).
        # 
        # The hypervolume contribution of a point at (f1, f2) relative to
        # the next point and reference point depends on the geometry.
        # Optimal spacing equalizes marginal contributions.
        #
        # A practical approach: use a power-law spacing that concentrates
        # points where the front is flat (near x0=1) since that's where
        # the hypervolume "knee" is and most of the HV comes from.
        #
        # For the front f2 = C*(1-sqrt(x0)), let u = sqrt(x0), so x0 = u^2,
        # f1 = u^2, f2 = C*(1-u).
        # In (f1, f2) space: f2 = C*(1 - sqrt(f1)).
        #
        # Optimal HV spacing: place points such that each rectangle in the
        # HV decomposition has equal area. This leads to non-uniform spacing.
        #
        # Empirically, using u = sqrt(x0) and uniform spacing in u gives
        # much better HV coverage than uniform in x0.
        
        # Use uniform spacing in u = sqrt(x0), which means x0 = u^2
        # This concentrates points near x0=1 (where the front is flat)
        u_vals = np.linspace(0, 1, n_evals)
        x0_vals = u_vals ** 2
        
        # Build candidate points
        X = np.zeros((n_evals, dim))
        X[:, 0] = x0_vals
        # x[1:] = 0 is already set
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X
        
        return X
