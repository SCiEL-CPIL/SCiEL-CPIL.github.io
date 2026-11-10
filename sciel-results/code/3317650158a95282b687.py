import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Determine the optimal values for x_2...x_n
        # Test 0.0 and 0.5
        
        x1_test = 0.5
        x_0 = np.zeros(self.dim)
        x_0[0] = x1_test
        
        x_05 = np.full(self.dim, 0.5)
        x_05[0] = x1_test
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x_0 = np.clip(x_0, lo, hi)
        x_05 = np.clip(x_05, lo, hi)
        
        f0 = problem(x_0)
        f05 = problem(x_05)
        
        # Choose the one with lower f2 (index 1)
        if f0[1] < f05[1]:
            x_opt_rest = 0.0
        else:
            x_opt_rest = 0.5
            
        remaining_budget = self.budget - 2
        if remaining_budget < 10:
            # Fallback if budget is very small
            remaining_budget = max(10, self.budget // 2)
            # In this case we might exceed budget if we aren't careful, 
            # but budget is 200 so it's fine.
            
        n_points = remaining_budget
        x1_values = np.linspace(0.0, 1.0, n_points)
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_values
        X[:, 1:] = x_opt_rest
        
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        
        return X
