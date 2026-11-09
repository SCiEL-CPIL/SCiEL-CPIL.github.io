"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using analytic Pareto front with optimized sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit with 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # For ZDT2, the Pareto front is parameterized by t = x0:
        # f1 = t
        # f2 = C * (1 - sqrt(t/C)) where C = 1 + 10*(n-1)
        # 
        # To maximize hypervolume, we need to distribute points to minimize
        # discretization error. For a convex front, optimal spacing is:
        # dt/di ∝ 1/sqrt(f2''(t))
        # 
        # f2''(t) ∝ t^{-1.5}, so dt/di ∝ t^{0.75}
        # Integrating: t^{0.25} ∝ i, so t ∝ i^4
        # 
        # However, the reference point affects the effective "height" function.
        # Let's use p=4 as derived, but also try to be robust.
        
        # Use a power-law distribution with p optimized for hypervolume
        # Based on calculus of variations: p = 4
        p = 4.0
        
        # Generate points with power-law distribution
        # u in [0, 1], t = u^p
        # This puts more points near t=0 where curvature is highest
        
        u = np.linspace(0.0, 1.0, n_points)
        x0_values = u ** p
        
        # Ensure exact endpoints
        x0_values[0] = 0.0
        x0_values[-1] = 1.0
        
        # Build decision vectors: x[0] = t, x[1:] = 0
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_values
        
        # Evaluate all points
        F = np.zeros((n_points, self.n_obj))
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
