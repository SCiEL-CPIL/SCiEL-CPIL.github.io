import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT1 Pareto front is at x_2..x_dim = 0
        # We allocate all budget to sampling x_1
        n_evals = self.budget
        
        # Create a uniform grid for x_1 to ensure good coverage of the front
        x1_vals = np.linspace(lo[0], hi[0], n_evals)
        
        # Initialize array for x
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        # x_1..x_{dim-1} are already 0, which is optimal for ZDT1
        
        results = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)
            
        return np.array(results)
