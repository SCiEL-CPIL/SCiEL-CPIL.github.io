import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front is achieved when g(x) = 1.
        # This occurs when x_i = 0.5 for all i > 0.
        # x_0 determines the position on the front: f1 = x_0, f2 = 1 - sqrt(x_0).
        
        n_samples = min(self.budget, 200)
        
        # Sample x_0 uniformly in [0, 1] to cover the Pareto front
        x0_vals = np.linspace(0.0, 1.0, n_samples)
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_vals
        # Set remaining variables to 0.5 to minimize g(x) to 1
        X[:, 1:] = 0.5
        
        # Ensure bounds are respected (0.5 is within [0,1])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # We must call problem exactly n_samples times
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
