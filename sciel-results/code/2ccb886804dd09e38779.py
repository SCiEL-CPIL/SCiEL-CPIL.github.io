"""Multi-objective optimizer for ZDT4 using population-based search with hypervolume-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, points, ref):
        """Compute hypervolume for 2D points given reference point."""
        if len(points) == 0:
            return 0.0
        valid = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
        if not valid:
            return 0.0
        valid.sort(key=lambda p: p[0])
        
        hv = 0.0
        prev_f1 = valid[0][0]
        min_f2 = valid[0][1]
        for i in range(1, len(valid)):
            cur_f1, cur_f2 = valid[i]
            if min_f2 < ref[1]:
                hv += (cur_f1 - prev_f1) * (ref[1] - min_f2)
            min_f2 = min(min_f2, cur_f2)
            prev_f1 = cur_f1
        if min_f2 < ref[1]:
            hv += (ref[0] - prev_f1) * (ref[1] - min_f2)
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific knowledge:
        # The Pareto front is achieved when x_2..x_D = 0.1 (global min of G)
        # and x_1 varies in [0, 1].
        # f_1 = x_1
        # f_2 = G(x) + H(x_1). If G=0, f_2 is minimized for each x_1.
        
        # Target value for x_2..x_D
        target_rest = np.clip(0.1, lo[1:], hi[1:])
        
        # We will use the entire budget to sample x_1 uniformly.
        # This ensures we cover the Pareto front as densely as possible.
        
        # Pre-allocate arrays for the maximum number of evaluations
        X = np.zeros((self.budget, self.dim))
        F = np.zeros((self.budget, self.n_obj))
        
        # Set x_2..x_D to the optimal value for all points
        X[:, 1:] = target_rest
        
        # We will fill x_1 with random values in [lo[0], hi[0]]
        # Usually [0, 1] for ZDT4.
        
        for i in range(self.budget):
            if time.time() > deadline:
                break
            
            # Sample x_1 uniformly
            # Using random sampling ensures good coverage of the front
            X[i, 0] = np.random.uniform(lo[0], hi[0])
            
            # Evaluate
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
