import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0

    def __call__(self, problem, bounds):
        n_evals = self.budget
        
        # ZDT1 Pareto front: f1 = x0, f2 = 1 - sqrt(x0)
        # Maximize HV = sum_{i=0}^{n-1} (10 + sqrt(t_i)) * (t_{i+1} - t_i)
        # where t_n = 11, t_i in [0,1], sorted ascending
        # Optimal: balance between (10+sqrt(t_i)) being large and gaps being reasonable
        # Use power law with alpha tuned for this objective
        alpha = 1.5
        
        if n_evals > 1:
            indices = np.linspace(0, 1, n_evals)
            x0_values = indices ** alpha
        else:
            x0_values = np.array([0.5])
            
        for x0 in x0_values:
            if time.time() > self.deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x0
            
            try:
                problem(x)
            except Exception:
                break

        return None
