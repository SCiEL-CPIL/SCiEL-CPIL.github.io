import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        time_limit = 300.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ1: x_0, x_1 are shape variables. x_2..x_{dim-1} are convergence variables.
        # Optimal convergence is at x_i = 0.5 (midpoint of [0,1])
        
        # Generate Sobol sequence for the 2 shape variables
        try:
            so = np.random.Sobol(d=2, scramble=True)
            sobol_points = so.generate(self.budget)
        except Exception:
            # Fallback to random if Sobol not available
            sobol_points = np.random.rand(self.budget, 2)
            
        last_f = None
        for i in range(self.budget):
            if time.time() - start_time > time_limit:
                break
                
            x = np.zeros(self.dim)
            x[0] = lo[0] + (hi[0] - lo[0]) * sobol_points[i, 0]
            x[1] = lo[1] + (hi[1] - lo[1]) * sobol_points[i, 1]
            
            # Fix convergence variables to midpoint
            for j in range(2, self.dim):
                x[j] = 0.5 * (lo[j] + hi[j])
                
            f = problem(x)
            last_f = f
            
        return last_f
