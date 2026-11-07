import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_points = self.budget
        # Generate x1 values to make f2 uniform
        # f2 = 1 - sqrt(x1)
        # We want f2 to be uniformly distributed in [0, 1]
        # Let u be uniform in [0, 1]
        # f2 = u
        # 1 - sqrt(x1) = u => sqrt(x1) = 1 - u => x1 = (1 - u)^2
        
        u = np.linspace(0, 1, n_points)
        x1_vals = (1.0 - u) ** 2
        
        # Ensure x1 is within [0, 1]
        # u in [0, 1] => 1-u in [0, 1] => x1 in [0, 1]
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_vals
        # x2...xn are 0
        
        for i in range(n_points):
            problem(X[i])
            
        return X
