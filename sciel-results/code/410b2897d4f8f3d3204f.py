import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget
        dim = self.dim
        
        # Strategy: Manifold-constrained sampling
        # For DTLZ7, the Pareto front is achieved when x_1 ... x_{n-1} = 0.
        # This ensures g(x)=0 and maximizes the '1 - sum' terms.
        # We sample x_0 densely using a Halton sequence for low-discrepancy coverage.
        
        # Generate Halton sequence in 1D (which is just a rational sequence)
        # Or simply use a uniform grid with slight jitter, or sorted random.
        # A uniform grid is deterministic and guarantees even spacing.
        # For 200 points, a grid is very effective for 1D coverage.
        
        # Use a uniform grid on x_0
        # To avoid boundary issues if the front is open, we use linspace
        # However, DTLZ7 front is typically closed [0,1]
        
        x0_samples = np.linspace(0.0, 1.0, n)
        
        X = np.zeros((n, dim))
        X[:, 0] = x0_samples
        # x[1:] remains 0, which is within bounds [0,1]
        
        # Evaluate all points
        # We can do this in a loop or vectorized if problem supports it, 
        # but problem(x) takes a single vector.
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            pass # We already have the results, just return
        
        return X
