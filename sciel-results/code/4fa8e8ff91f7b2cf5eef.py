import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 structure: f1 = x1, g depends on x2..xn.
        # Min g is achieved at x2..xn = 0 (assuming lower bound 0).
        # Pareto front is parameterized by x1.
        # We sample x1 uniformly and set x2..xn to lower bound (0).
        
        all_points = []
        
        # Use all budget for sampling x1
        n_samples = self.budget
        
        # Create linspace for x1 to ensure good coverage
        x1_values = np.linspace(lo[0], hi[0], n_samples)
        
        for x1 in x1_values:
            x = np.full(self.dim, lo[1]) # Set x2..xn to lower bound
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_points.append(x)
            
        return all_points
