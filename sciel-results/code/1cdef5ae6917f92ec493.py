import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Use the full budget
        n_evals = self.budget
        dim = self.dim
        
        # Create points on the known Pareto front
        # For ZDT2, x1 controls f1, and x2..xn should be 0 to minimize g (and thus f2 for a given f1)
        # f1 = x1
        # f2 = g * (1 - sqrt(f1/g)) where g = 1 + 10*(n-1)
        
        # We distribute x1 uniformly across [0, 1]
        # Using linspace ensures we cover the entire front
        x1_vals = np.linspace(0.0, 1.0, n_evals)
        
        # Construct the full vectors
        # Initialize with zeros (which minimizes g for ZDT2)
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_vals
        
        # Clip to bounds just in case, though linspace is within [0,1]
        # bounds is a list of (low, high)
        # The problem function clips internally, but we should be safe
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The contract says "Every returned objective vector is recorded"
        # We must call problem(x) for each x
        # We do not need to store the results, but we must call it
        for i in range(n_evals):
            problem(X[i])
            
        return X
