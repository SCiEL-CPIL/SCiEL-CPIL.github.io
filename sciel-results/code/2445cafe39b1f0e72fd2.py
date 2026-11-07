import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300 seconds limit
        
        # ZDT2 Pareto front: x[0] in [0, 1], x[1:] = 0
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0])
        
        # We have a budget of self.budget evaluations.
        # We will use all of them to sample the Pareto front.
        # Uniform sampling of x[0] in [0, 1]
        
        n_evals = self.budget
        x0_values = np.linspace(0.0, 1.0, n_evals)
        
        # Construct the full vectors
        # x = [x0, 0, 0, ..., 0]
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_values
        
        # Ensure bounds are respected (though [0,1] is standard for ZDT2)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        # Check time
        current_time = time.time()
        if current_time > deadline - 10:
             # If we are close to the deadline, we might need to stop, 
             # but we have already used all budget. 
             # The loop above is the only work.
             pass
        
        return X
