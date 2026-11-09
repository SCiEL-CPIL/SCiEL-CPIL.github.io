import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit per task, use 300s to be safe
        
        # 1. Setup
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 2. Fix variables 3..11 to midpoint
        mid = (lo + hi) / 2.0
        fixed_vars = mid[3:]
        
        # 3. Generate grid for first 3 variables
        # Budget is 200.
        # Strategy: Denser grid on first 2 dims (which likely map to objectives 1 and 2 directly)
        # and slightly less dense on the 3rd.
        # Or use the knowledge that WFG9 has a specific structure.
        # For WFG9, the Pareto front is defined by the first 2 variables.
        # Let's try a denser grid: 10 * 10 * 2 = 200? No, that's too coarse on dim 2.
        # Let's try 10 * 5 * 4 = 200.
        # Or 8 * 5 * 5 = 200.
        # The previous score was 1.0348 with 5*5*8.
        # Let's try to exploit the structure better.
        # In WFG9, x1 and x2 determine the position on the front. x3...x12 affect convergence.
        # If we fix x3...x12 to optimal values for convergence, we only need to sweep x1 and x2.
        # What are the optimal values for x3...x12?
        # For WFG9, the ideal point is (1,1,1).
        # The convergence variables are x3...x12.
        # Usually, setting them to 0.5 or some specific value helps.
        # Let's assume the current midpoint is decent, but maybe we can do better with a local search on the convergence variables for a few points.
        
        # Given the time constraint and the "deterministic grid" directive, let's stick to grid but optimize the shape.
        # 10 * 10 * 2 = 200.
        # 10 * 5 * 4 = 200.
        # 8 * 8 * 3 = 192 (leaves 8 for local search?)
        
        # Let's try 10 * 10 * 2 = 200. This gives better resolution on the first two dimensions.
        n0, n1, n2 = 10, 10, 2
        
        # Create linspace for each of the first 3 dims
        # Include endpoints to ensure full front coverage
        xs0 = np.linspace(lo[0], hi[0], n0)
        xs1 = np.linspace(lo[1], hi[1], n1)
        xs2 = np.linspace(lo[2], hi[2], n2)
        
        # Meshgrid
        G0, G1, G2 = np.meshgrid(xs0, xs1, xs2, indexing='ij')
        
        # Flatten
        X0 = G0.flatten()
        X1 = G1.flatten()
        X2 = G2.flatten()
        
        # Construct full vectors
        # X is (N, dim)
        N = len(X0)
        X = np.zeros((N, self.dim))
        X[:, 0] = X0
        X[:, 1] = X1
        X[:, 2] = X2
        X[:, 3:] = fixed_vars
        
        # Evaluate
        results = []
        for i in range(N):
            if time.time() > deadline - 10: # Safety margin
                break
            f = problem(X[i])
            results.append(f)
            
        # If we have time and budget left, do a local search on the non-dominated points
        # But first, let's just return the grid results. The grid itself might be sufficient if well-shaped.
        
        return X # Return X as per signature, though return value is ignored
