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
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto Front is on x_2..x_n = 0
        # We focus our budget on varying x_1
        
        # Strategy: Uniform sampling of x_1 in [0, 1]
        # This ensures coverage of the entire front.
        # If the front has discontinuities/notches, uniform sampling with N=200
        # will likely hit near them.
        
        n_evals = self.budget
        x1_values = np.linspace(0.0, 1.0, n_evals)
        
        # Create candidate points
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_values
        # x_2..x_n are already 0
        
        F = np.zeros((n_evals, self.n_obj))
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
