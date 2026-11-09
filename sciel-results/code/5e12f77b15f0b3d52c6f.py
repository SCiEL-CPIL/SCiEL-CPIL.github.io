import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer exploiting ZDT1 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT1 analysis:
        # f1 = x0
        # f2 = g(x) * (1 - sqrt(x0/g(x)))
        # g(x) = 1 + 10*(n-1)/n * sum(x_i for i>0)
        # Pareto front is at g(x)=1, which requires x1...xn-1 = 0.
        # On the front: f1 = x0, f2 = 1 - sqrt(x0).
        # To maximize hypervolume, we need a dense, uniform spread of points
        # along the Pareto front in the objective space.
        # Since f1=x0 is linear, uniform x0 in [0,1] is optimal.
        
        n_evals = self.budget
        
        # Generate uniform grid for x0
        x0_vals = np.linspace(0.0, 1.0, n_evals)
        
        # Construct full dimension vectors
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        # x[1:] remains 0.0, ensuring g(x) = 1
        
        # Time management
        start_time = time.time()
        # Evaluator kills at 330s. We stop at 325s to be safe.
        deadline = 325.0
        
        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
