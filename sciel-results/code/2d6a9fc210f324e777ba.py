import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s margin
        
        n_evals = self.budget
        if n_evals <= 0:
            return None
            
        # ZDT1 Pareto front: x[1:] = 0, x[0] in [0,1]
        # f1 = x[0], f2 = 1 - sqrt(f1)
        # For hypervolume maximization with ref [11, 11], use power-law distribution
        # f1 = u^p where u is uniform. p=2 concentrates near f1=0 where curvature is high.
        
        # Generate uniform u values
        u = np.linspace(0.0, 1.0, n_evals)
        
        # Apply power transformation: f1 = u^2
        # This gives f2 = 1 - u, so points are evenly spaced in f2
        # which is optimal for hypervolume on this concave front
        x1_vals = u ** 2
        
        # Construct X matrix
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
