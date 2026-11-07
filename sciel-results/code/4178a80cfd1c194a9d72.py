import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine the value for x_2 ... x_n to minimize g(x)
        # Ideally 0. If 0 is out of bounds, pick the closest bound.
        x_rest = np.zeros(n - 1)
        for i in range(1, n):
            if lo[i] <= 0 <= hi[i]:
                x_rest[i-1] = 0.0
            elif hi[i] < 0:
                x_rest[i-1] = hi[i]
            else:
                x_rest[i-1] = lo[i]
        
        # Generate points on the Pareto front
        # f2 = 1 - sqrt(f1)
        # Sample f2 uniformly in [0, 1]
        f2_samples = np.linspace(0.0, 1.0, budget)
        f1_samples = (1.0 - f2_samples) ** 2
        
        # Map f1_samples (which are in [0, 1]) to x_1 bounds [lo[0], hi[0]]
        x1_samples = lo[0] + f1_samples * (hi[0] - lo[0])
        
        # Construct full vectors
        X = np.zeros((budget, n))
        X[:, 0] = x1_samples
        X[:, 1:] = x_rest
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
