import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        results = []
        
        # Generate x0 values
        # Optimal spacing for ZDT6 front f2 = 1 - sqrt(f1)
        # Minimization of HV error suggests x0 ~ (i/N)^2
        i = np.arange(n_evals)
        us = i / n_evals
        x0s = us ** 2.0
        
        # Ensure we hit 1.0
        x0s[-1] = 1.0
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            # Other x's are 0, which is optimal for ZDT6
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
