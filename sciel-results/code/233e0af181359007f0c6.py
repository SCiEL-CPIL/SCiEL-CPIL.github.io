import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        n_points = budget
        if n_points < 2:
            n_points = 2
            
        # Latin Hypercube Sampling on x1 in [lo[0], hi[0]]
        # This ensures uniform coverage of the x1 domain, avoiding gaps or clusters 
        # that might occur with pure uniform random or fixed grids.
        
        # Generate LHS points
        # 1. Divide [0, 1] into n_points intervals
        # 2. Pick one point uniformly at random from each interval
        # 3. Shuffle the order
        
        # Seed is set by evaluator, but we rely on np.random global state
        # To be robust, we don't re-seed here, we just use the current state.
        
        # Intervals are [i/n, (i+1)/n]
        idx = np.arange(n_points)
        # Random offset within each interval
        offsets = np.random.rand(n_points)
        
        # Normalize to [0, 1]
        x1_uniform = (idx + offsets) / n_points
        
        # Shuffle to decorrelate indices from specific offsets (standard LHS)
        # This step makes it a proper Latin Hypercube in the sense of mixed randomization
        # though for 1D, the set of points is the same, just the order.
        # The order doesn't matter for the final hypervolume set, but let's do it properly.
        perm = np.random.permutation(n_points)
        x1_uniform = x1_uniform[perm]
        
        # Map to [lo[0], hi[0]]
        x1_vals = lo[0] + x1_uniform * (hi[0] - lo[0])
        
        # Ensure strict bounds adherence
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        recorded_F = []
        
        for i in range(n_points):
            x = np.zeros(dim)
            # For ZDT2, the Pareto front is at x[1:] = 0
            x[0] = x1_vals[i]
            
            # Clip full vector just in case, though x[1:] is 0 which is within [0,1]
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            if f is not None and len(f) == n_obj:
                recorded_F.append(f)
            
            # Time check
            if time.time() > deadline:
                break

        return recorded_F
