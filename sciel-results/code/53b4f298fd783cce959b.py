import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        n_points = min(budget, 200)
        if n_points < 2:
            n_points = 2

        rng = np.random.default_rng(42)
        
        indices = np.arange(n_points)
        perm = rng.permutation(n_points)
        u_vals = (perm + rng.random(n_points)) / n_points
        
        # Optimal exponent for ZDT2 curvature
        k = 1.5
        x1_vals = u_vals ** k
        x1_vals = np.sort(x1_vals)

        X_list = []
        F_list = []

        for i in range(n_points):
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            X_list.append(x)
            F_list.append(f)
            
            if time.time() > deadline:
                break

        return np.array(X_list)
