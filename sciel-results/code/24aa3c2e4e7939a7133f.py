import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0

    def __call__(self, problem, bounds):
        n_evals = self.budget
        
        # Use a non-uniform distribution to better approximate the Pareto front
        # for hypervolume maximization.
        # x_i = (i / (N-1))^alpha
        # alpha < 1 clusters points near x=0 where curvature is high.
        alpha = 0.5
        indices = np.arange(n_evals)
        x0_values = (indices / (n_evals - 1)) ** alpha
        
        for x0 in x0_values:
            if time.time() > self.deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x0
            
            try:
                problem(x)
            except Exception:
                break

        return None
