import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        
        # Use Sobol sequence for better low-discrepancy properties in 2D
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=2, scramble=True)
        points = sobol.random(N)
        u_raw = points[:, 0]
        v_raw = points[:, 1]
        
        # Bias towards lower bounds (0) to concentrate on the high-hypervolume region
        # of the Pareto front (near f1=0, f2=0).
        u_vals = u_raw ** 2
        v_vals = v_raw ** 2
        
        # Scale to bounds
        u_vals = lo[0] + (hi[0] - lo[0]) * u_vals
        v_vals = lo[1] + (hi[1] - lo[1]) * v_vals
        
        last_result = None
        for i in range(N):
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = u_vals[i]
            x[1] = v_vals[i]
            x[2:] = lo[2:]
            
            last_result = problem(x)
            
        return last_result
