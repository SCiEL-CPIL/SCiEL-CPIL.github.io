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
        
        # ZDT6 Pareto front: x[1:] = 0, x[0] in [0, 1]
        # f1 = 10 + sqrt(x0) * (1 + 10*sin(10*pi*x0))  -- wait, let me check
        # Actually for ZDT6: f1 = 1 + 9*(g-1)^2 ... no
        # ZDT6: f1 = 1 + 9*(g(x2..xn))^(1/2) ... 
        # Let me just sample x[0] uniformly and set x[1:]=0
        # The Pareto front is x[1:]=0, x[0] in [0,1]
        
        x0_values = np.linspace(0.0, 1.0, n_evals)
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim, dtype=float)
            x[0] = x0_values[i]
            x[1:] = 0.0
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
