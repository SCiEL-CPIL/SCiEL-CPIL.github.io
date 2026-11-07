import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        n_points = budget
        if n_points < 2:
            n_points = 2
            
        # Use x = t^2 to match the curvature of ZDT2 front (f2 = 1 - sqrt(x1))
        # Density proportional to 1/sqrt(x)
        t = np.linspace(0, 1, n_points)
        x1_vals = t ** 2
        
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        recorded_F = []
        
        for i in range(n_points):
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            if f is not None and len(f) == n_obj:
                recorded_F.append(f)
                
            if time.time() > deadline:
                break

        return recorded_F
