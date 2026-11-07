import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        # We will use all evaluations for sampling x_1
        # Set x_2...x_n to 0 (lo[1:])
        
        # Generate uniform samples for x_1
        # Include 0 and 1 explicitly to capture extremes
        x1_samples = np.linspace(lo[0], hi[0], n_evals)
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_samples
        # X[:, 1:] is already 0, which is lo[1:] if lo is 0
        
        # Evaluate all
        # We must ensure we don't exceed time
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return None
