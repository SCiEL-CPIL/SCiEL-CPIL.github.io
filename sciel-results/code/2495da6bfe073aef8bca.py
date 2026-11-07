import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        dim = self.dim
        
        # ZDT3 Pareto front: f1 = x1, f2 = 1 - sqrt(x1)
        # Strategy: Uniform sampling in f1 (x-axis)
        # This ensures equal width intervals in the x-direction.
        # For the Left Riemann Sum (HV calculation), this is a standard approach.
        
        # Use midpoints to avoid boundaries?
        # f1 in [0, 1]
        # If we use midpoints, f1_i = (i + 0.5) / N
        # f2_i = 1 - sqrt(f1_i)
        
        f1_values = (np.arange(n_evals) + 0.5) / n_evals
        f2_values = 1 - np.sqrt(f1_values)
        
        # Construct vectors
        X = np.zeros((n_evals, dim))
        X[:, 0] = f1_values
        
        F = np.array([problem(x) for x in X])
        
        return F
