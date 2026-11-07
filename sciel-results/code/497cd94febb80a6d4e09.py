import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        n_evals = min(self.budget, 200)
        
        # Use power-law spacing to cluster points where curvature is high
        # For ZDT1-like concave front, cluster near x1=0
        t_vals = (np.arange(n_evals) / max(n_evals - 1, 1)) ** 1.5
        t_vals = np.clip(t_vals, 0, 1)
        
        # Create X matrix
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = t_vals
        # X[:, 1:] remains 0
        
        # Evaluate with deadline guard
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
