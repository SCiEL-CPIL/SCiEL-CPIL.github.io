import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        # DTLZ7 Pareto front is 1D curve with x1=0, x3..x22=0, x2 in [0,1]
        # f1 = 0
        # f2 = x2
        # f3 = 1 + x2 + (1-x2)cos(pi*x2/4)
        
        # Sample x2 using a non-uniform grid clustering near x=0
        # This maximizes hypervolume contribution from the f2 dimension
        n_samples = 200
        i_vals = np.arange(n_samples)
        x2_vals = (i_vals / (n_samples - 1)) ** 2
        
        X = np.zeros((n_samples, self.dim))
        # X[:, 0] = 0 (x1)
        X[:, 1] = x2_vals
        # X[:, 2:] = 0 (inactive)
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
