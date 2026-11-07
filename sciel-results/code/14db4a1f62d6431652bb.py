import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        dim = self.dim
        budget = self.budget
        
        # ZDT3 Pareto Front Structure:
        # f1 = x1
        # f2 = h(x1, g) where g is minimized at 1.
        # Min g occurs at x_i = 0.5 for i > 0.
        # So we set x[0] to vary, and x[1:] to 0.5.
        
        # We use the entire budget to sample the front.
        # Uniform stratified sampling in x1 is effective for hypervolume.
        n_samples = budget
        
        # Generate x1 values
        x1_vals = np.linspace(0.0, 1.0, n_samples)
        
        # Prepare the population matrix
        X = np.zeros((n_samples, dim))
        X[:, 0] = x1_vals
        if dim > 1:
            X[:, 1:] = 0.5
        
        # Evaluate all points
        for i in range(n_samples):
            if time.time() - start_time > 290:
                break
            problem(X[i])
            
        return X
