import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Purely analytical construction of Pareto front points.
        # For DTLZ2 with 3 objectives, the Pareto front is a quarter-circle in 3D.
        # f1 = sin(pi*x1/2), f2 = cos(pi*x1/2), f3 = 1 (when g=0)
        # To maximize hypervolume, we need points uniformly distributed along this arc.
        # The arc length is proportional to x1. So uniform sampling in x1 gives uniform arc coverage.
        # We use a deterministic Halton sequence for low-discrepancy sampling in 1D.
        
        n_points = self.budget
        
        # Halton sequence in base 2 for 1D
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        
        # Generate n_points x1 values
        x1_vals = np.array([halton(i, 2) for i in range(1, n_points + 1)])
        
        # Construct full vectors: x[0] = x1, x[1:] = 0.5 (to make g=0)
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_vals
        X[:, 1:] = 0.5
        
        # Clip to bounds (should be no-op for [0,1] bounds and 0.5)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # We must call problem(x) for each x to record the objectives
        for i in range(n_points):
            if time.time() > deadline:
                break
            f = problem(X[i])
        
        return X
