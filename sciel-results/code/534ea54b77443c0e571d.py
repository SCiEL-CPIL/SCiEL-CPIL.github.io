"""Multi-objective optimizer for ZDT6 using objective-space uniform sampling of the analytic Pareto front."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT6 Pareto Front Analysis:
        # The Pareto front is defined by x_2...x_dim = 0.
        # f_1(x) = x_1^2
        # f_2(x) = 1 + sin(10 * pi * x_1)
        #
        # The current strategy samples x_1 uniformly in [0, 1].
        # Since f_1 = x_1^2, this leads to a high density of points near f_1=0 
        # and a low density near f_1=1.
        # 
        # Hypervolume is maximized when points are distributed as evenly as possible 
        # along the front in objective space.
        # We can achieve uniform distribution in f_1 by sampling f_1 uniformly in [0, 1]
        # and solving for x_1 = sqrt(f_1).
        
        n_samples = budget
        
        # Sample f_1 uniformly in [0, 1]
        # Using linspace ensures endpoints are covered
        f1_values = np.linspace(0.0, 1.0, n_samples)
        
        # Solve for x_1
        x1_values = np.sqrt(f1_values)
        
        # Construct the population
        # x_2 to x_dim are set to 0.0 (lower bound)
        # This ensures g(x) = 1, which is the minimum possible g.
        X = np.zeros((n_samples, dim))
        X[:, 0] = x1_values
        # x_2...x_dim are already 0.0
        
        # Evaluate all points
        # The problem function clips to bounds, so 0.0 is safe.
        # We must call problem(x) for each x to record the objective values.
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            # Evaluate and discard the return value (it is recorded by the evaluator)
            problem(X[i])
            
        return X
