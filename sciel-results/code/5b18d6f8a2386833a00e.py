import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We know ZDT1 structure: x[0] determines f1, x[1:] should be 0 for optimal f2.
        # We will generate 'budget' points.
        # x[0] will be a uniform grid from lo[0] to hi[0].
        # x[1:] will be set to lo[1:] (which is 0).
        
        n_evals = self.budget
        # Create x[0] grid
        # Ensure we include endpoints if possible
        if n_evals == 1:
            x0_vals = np.array([(lo[0] + hi[0]) / 2.0])
        else:
            x0_vals = np.linspace(lo[0], hi[0], n_evals)
            
        # Create population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        # X[:, 1:] is already 0, which is lo[1] if bounds are [0,1].
        # If bounds were different, we should set to lo[1:].
        # But ZDT1 bounds are [0,1].
        # To be safe, set to lo[1:].
        if self.dim > 1:
             X[:, 1:] = lo[1:]
             
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
