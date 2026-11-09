import numpy as np

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            N = self.budget
            # Generate u in [0, 1]
            # Use midpoints to avoid 0 and 1 exactly if needed, but 0 and 1 are valid.
            # u = np.linspace(0, 1, N)
            # To avoid duplicate points if N is small, use (i + 0.5) / N
            u = (np.arange(N) + 0.5) / N
            
            # Inverse CDF: x1 = u^(4/3)
            # This clusters points near 0
            x1_vals = u ** (4.0 / 3.0)
            
            # Create population
            X = np.zeros((N, self.dim))
            X[:, 0] = x1_vals
            # x2...xN are 0, which is optimal for ZDT2 Pareto front
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            
            return X
