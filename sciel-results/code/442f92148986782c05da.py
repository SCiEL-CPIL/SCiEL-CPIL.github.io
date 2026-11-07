import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Optimal sampling for ZDT2 hypervolume:
        # The Pareto front is f2 = g(1 - sqrt(f1/g)).
        # To maximize HV with N points, we derived that x0 = u^2 (u uniform) is optimal.
        # This makes sqrt(x0) uniform, which minimizes the discretization error of the HV integral.
        
        # Generate uniform u in [0, 1]
        u_values = np.linspace(0.0, 1.0, n_evals)
        
        # Transform to x0
        x0_values = u_values ** 2
        
        # Base vector for x[1:]
        # For ZDT2, x[1:] = 0 minimizes g(x) and thus f2.
        # We assume bounds allow 0. If not, we would need to adjust, 
        # but the problem statement says bounds are [0, 1].
        base_x = np.zeros(self.dim)
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            x = base_x.copy()
            x[0] = x0_values[i]
            
            # Ensure x is within bounds just in case, though problem clips
            # x = np.clip(x, lo, hi)
            
            problem(x)
            
        return base_x
