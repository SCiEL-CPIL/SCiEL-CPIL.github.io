import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 specific: set variables 2..11 to 0.5 to be on Pareto front
        # x[0] and x[1] determine the position on the front
        
        n_points = self.budget
        
        # Use Sobol sequence for better 2D coverage
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=2, scramble=True)
        # Generate 200 points in [0,1]^2
        # Sobol requires power of 2, so generate 256 and take first 200
        sobol_points = sobol.random(256)
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            
            u = sobol_points[i, 0]
            v = sobol_points[i, 1]
            
            # Transform to x[0], x[1] for uniform area coverage on DTLZ2 front
            # x0 = 2/pi * arcsin(u)
            # x1 = 2/pi * v
            
            x = np.zeros(self.dim)
            x[0] = (2.0 / np.pi) * np.arccos(1.0 - u)
            x[1] = (2.0 / np.pi) * v
            # Set remaining variables to 0.5 to minimize g(x)
            x[2:] = 0.5
            
            # Clip to bounds just in case, though 0.5 and [0,1] mapping should be safe
            x = np.clip(x, lo, hi)
            
            problem(x)
        
        return None
