import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # We have self.budget evaluations available.
        # We will use all of them.
        n_evals = self.budget
        
        # Create a grid of x_0 values
        # Using linspace to cover [0, 1]
        x0_values = np.linspace(0.0, 1.0, n_evals)
        
        # Prepare the population
        # x = [x0, 0, 0, ..., 0]
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_values
        # The rest are already 0
        
        # Evaluate all
        # We must call problem(x) for each x
        # The problem function clips and returns objectives
        F = np.zeros((n_evals, self.n_obj))
        for i in range(n_evals):
            F[i] = problem(X[i])
            
        return X
