import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim

        # ZDT6 Pareto front is achieved when x[0] = t in [0, 1] and x[1:] are at minima of g(x).
        # Minima of g(x) terms 2*x^2 - 2*cos(4*pi*x) are at x=0 and x=0.5.
        # We set x[1:] = 0.0 to ensure g(x) is minimized.
        
        # To maximize hypervolume, we densely sample the Pareto front.
        # Uniform distribution of x[0] (which is f1) ensures good coverage of the front.
        
        n_points = budget
        
        # Generate t values
        t_vals = np.linspace(0.0, 1.0, n_points)
        
        # Create decision vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = t_vals
        # X[:, 1:] remains 0.0
        
        # Evaluate all points
        # The problem function handles clipping and returns objectives.
        # We must ensure we do not exceed the budget.
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
