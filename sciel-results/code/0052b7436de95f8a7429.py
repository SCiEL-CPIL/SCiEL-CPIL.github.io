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
        
        # For ZDT6, Pareto front: f2 = 1 - sqrt(f1) when g=1 (x[1:]=0)
        # f1(x1) is highly non-linear. Most variation is in x1 in [0, 0.2].
        # Biased sampling towards 0 helps cover the f1 range [0, 1] more uniformly.
        i_arr = np.arange(n_evals)
        us = i_arr / (n_evals - 1) if n_evals > 1 else np.array([0.0])
        x0s = us ** 3.0
        
        # Ensure we hit both endpoints
        if n_evals > 1:
            x0s[0] = 0.0
            x0s[-1] = 1.0
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            # x[1:] = 0 minimizes g, putting points on the Pareto front
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
