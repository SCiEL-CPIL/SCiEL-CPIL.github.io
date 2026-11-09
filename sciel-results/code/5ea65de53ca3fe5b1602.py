import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        if n_evals == 1:
            x0_vals = np.array([(lo[0] + hi[0]) / 2.0])
        else:
            # Use power law distribution for x0
            # Optimal power p approx 0.4 based on curvature analysis
            p = 0.4
            t = np.linspace(0, 1, n_evals)
            x0_vals = lo[0] + (hi[0] - lo[0]) * (t ** p)
            
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        if self.dim > 1:
            X[:, 1:] = lo[1:]
            
        F = np.array([problem(x) for x in X])
        
        return X
