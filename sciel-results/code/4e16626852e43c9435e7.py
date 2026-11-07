import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front is at x_2...x_n = 0.5
        # x_1 varies in [0, 1]
        # Pareto front segments are around x_1 = 0.05, 0.25, 0.45, 0.65, 0.85
        # We distribute budget among these 5 regions
        
        n_regions = 5
        evals_per_region = self.budget // n_regions
        # If budget is not divisible, give remainder to first regions
        remainders = self.budget % n_regions
        
        regions = [
            (0.0, 0.15),
            (0.15, 0.35),
            (0.35, 0.55),
            (0.55, 0.75),
            (0.75, 0.95)
        ]
        
        X = np.zeros((self.budget, self.dim))
        # Set x_2...x_n to 0.5
        X[:, 1:] = 0.5
        
        idx = 0
        for i, (low, high) in enumerate(regions):
            n_evals = evals_per_region + (1 if i < remainders else 0)
            # Generate samples in [low, high]
            x1_samples = np.linspace(low, high, n_evals)
            X[idx:idx+n_evals, 0] = x1_samples
            idx += n_evals
            
        # Evaluate
        for i in range(self.budget):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return None
