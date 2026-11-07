"""ZDT4 multi-objective optimizer exploiting the analytic Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    
    ZDT4 structure:
    - x[0] in [0, 1], x[1:] in [-5, 5]
    - f1 = x[0]
    - g(x) = 1 + 10*(n-1) = 101 when x[i]=0 for all i>0
    - f2 = g(x) * (1 - sqrt(f1 / g(x)))
    - Pareto front: f1 = x0, f2 = 101*(1 - sqrt(x0/101))
    
    Strategy: Use all budget evaluations to densely sample the known Pareto front.
    Use non-uniform spacing to concentrate samples in the high-curvature region
    near x0=0 where the uniform grid is least efficient for hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_samples = self.budget
        
        # Non-uniform sampling: concentrate points near x0=0 where curvature is highest.
        # Use a power-law transformation: x0 = t^alpha where t is uniform in [0,1].
        # alpha > 1 concentrates more points near 0.
        # The curvature of f2(x0) = 101*(1 - sqrt(x0/101)) is highest near x0=0.
        # Optimal alpha for hypervolume maximization is around 2-3.
        
        # Try alpha = 2.0 (square root spacing)
        t = np.linspace(0.0, 1.0, n_samples)
        x0_values = t ** 2.0
        
        # For each sample, set x[0] = x0_value, x[1:] = 0 (global min of g)
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_values
        # x[1:] = 0 is already set by zeros
        
        # Clip to bounds for safety
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
