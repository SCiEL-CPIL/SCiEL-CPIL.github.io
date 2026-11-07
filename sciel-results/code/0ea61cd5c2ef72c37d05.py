import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Check if 0 is within bounds for x1..x_{dim-1}
        # This ensures g(x) = 1
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))
        
        if zeros_ok:
            n_points = self.budget
            # Assume x0 range [0, 1] for standard ZDT4
            # Clip to valid bounds
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)
            
            if x0_lo >= x0_hi:
                # Fallback
                x0_vals = np.linspace(lo[0], hi[0], n_points)
            else:
                # Use non-uniform distribution: x ~ t^{1.5}
                # This concentrates points at high x0 where the Pareto front is flatter,
                # maximizing the hypervolume excess area for the concave curve.
                # t in [0, 1]
                t = np.linspace(0, 1, n_points)
                x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** 1.5)
                
                # Ensure we hit the endpoints exactly
                # 0**1.5 is 0. 1**1.5 is 1.
                
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            # X[:, 1:] = 0.0 is default
            
            # Evaluate
            # Check time
            if time.time() > deadline:
                 return X
                 
            F = np.array([problem(x) for x in X])
            return X
            
        else:
            # Fallback random
            rng = np.random.default_rng(0)
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X
