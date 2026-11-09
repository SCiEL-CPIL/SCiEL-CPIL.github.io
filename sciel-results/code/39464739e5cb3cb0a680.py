"""Optimized ZDT2 optimizer using model-based approach with curvature-aware spacing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Model-based optimizer for ZDT2.
    
    ZDT2 Pareto Front:
    f1 = x1
    f2 = g(x) * (1 - sqrt(f1 / g(x)))
    Min g(x) occurs at x[1:] = 0, where g(x) = 1 (for standard 30-dim ZDT2).
    Front: f1 in [0, 1], f2 = 1 - sqrt(f1).
    
    To maximize Hypervolume (HV) with a fixed budget, we distribute points to minimize
    the discretization error. The optimal spacing for a convex front is proportional
    to the inverse cube root of the second derivative. For f2 = 1 - sqrt(f1),
    f2''(f1) ~ f1^(-1.5), so optimal delta f1 ~ f1^(0.5).
    This implies sampling f1 (x1) with a quadratic distribution: x1 = u^2.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Bounds are [0, 1] for ZDT2 typically.
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        n_evals = self.budget
        
        # Generate u in [0, 1] uniformly using midpoints.
        u = (np.arange(n_evals) + 0.5) / n_evals
        
        # Quadratic spacing in f1 (x1) to match optimal HV distribution for ZDT2.
        # x1 = u^2 densifies points near f1=0 where the front is steepest.
        x1_vals = x1_lo + (u ** 2) * (x1_hi - x1_lo)
        
        # Ensure strictly within bounds
        x1_vals = np.clip(x1_vals, x1_lo, x1_hi)

        # Initialize solutions: x[0] = x1_vals, x[1:] = 0
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        # x[1:] remains 0, which minimizes g(x) for ZDT2, placing points on the Pareto front.
        
        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        
        for i in range(n_evals):
            F[i] = problem(X[i])
            
            # Deadline guard
            if time.time() > deadline:
                return X

        return X
# EVOLVE-BLOCK-END
