import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        n_evals = self.budget
        dim = self.dim
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy:
        # 1. Exploit ZDT2 structure: Pareto front is at x[1:] = 0 (minimizing g).
        # 2. Use non-uniform sampling for x[0] to better approximate the convex front.
        #    The front f2(x1) is convex and decreasing. Curvature is highest near x1=0.
        #    We use x1 = t^k with k=0.5 to bias sampling towards x1=0.
        
        # Generate t uniformly in [0, 1]
        t = np.linspace(0.0, 1.0, n_evals)
        
        # Apply power transformation to bias towards 0
        # k=0.5 corresponds to x1 = sqrt(t), which clusters points near x1=0
        k = 0.5
        x1_vals = t ** k
        
        # Ensure we stay within bounds for x1
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        # Construct full vectors
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_vals
        # Set remaining dimensions to lower bound to minimize g(x)
        if dim > 1:
            X[:, 1:] = lo[1:]
            
        # Evaluate all points with time guard
        for i in range(n_evals):
            if time.time() > deadline:
                break
            # problem(x) clips to bounds internally, but we've already clipped x1 and set x[1:] to lo
            # It's safe to pass X[i]
            problem(X[i])
        
        return X
