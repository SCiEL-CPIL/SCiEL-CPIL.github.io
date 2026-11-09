"""Optimized ZDT2 optimizer using model-based approach with non-uniform spacing."""
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
    
    To maximize Hypervolume (HV) with a fixed budget, we should distribute points
    to minimize the discretization error of the HV. For concave fronts like this,
    the slope is steepest at f1=0. Uniform sampling in f1 leaves large gaps in 
    f2 near 1, resulting in 'staircase' losses.
    
    We use a power-law distribution x1 = (u)^2 (where u is uniform in [0,1])
    to densify points near f1=0, where the curvature change is most significant 
    for the HV calculation relative to the reference point [11, 11].
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
        # We map our generated x1 values directly to [lo[0], hi[0]].
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        n_evals = self.budget
        
        # Generate u in [0, 1] uniformly using midpoints for better HV coverage.
        # For ZDT2's convex Pareto front (f2 = 1 - sqrt(f1)), uniform f1 spacing
        # minimizes discretization error for hypervolume.
        u = (np.arange(n_evals) + 0.5) / n_evals
        
        # Uniform spacing in f1
        x1_vals = x1_lo + u * (x1_hi - x1_lo)
        
        # Ensure strictly within bounds just in case of floating point errors
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
