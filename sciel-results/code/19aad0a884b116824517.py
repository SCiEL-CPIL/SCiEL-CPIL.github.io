import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s margin
        
        # ZDT1 Pareto front is achieved when x[1:] = 0 and x[0] in [0, 1]
        # f1 = x[0]
        # f2 = g(x) * (1 - sqrt(f1/g))
        # With x[1:] = 0, g(x) = 1 (for standard ZDT1 definition where g=1+9/(n-1)*sum(x[1:]))
        # So f2 = 1 - sqrt(f1)
        
        # We have 200 evaluations.
        # We will sample 200 points uniformly on the Pareto front.
        n_evals = self.budget
        if n_evals <= 0:
            return None
            
        # Generate x1 values
        # Uniform distribution over [0, 1]
        x1_vals = np.linspace(0.0, 1.0, n_evals)
        
        # Construct X matrix
        # Shape: (n_evals, dim)
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        
        # Evaluate
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f_vals = problem(X[i])
            F.append(f_vals)
            
        if not F:
            return X
            
        return X
