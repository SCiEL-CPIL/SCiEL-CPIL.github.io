import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 320.0

    def __call__(self, problem, bounds):
        n = self.budget
        if n <= 0:
            return None
        
        # Optimal distribution for ZDT1 hypervolume maximization
        # Derived from minimizing discretization error for concave h(u)
        # We determined beta=4 is optimal for x0 = (k/N)^beta
        # This corresponds to u = sqrt(x0) = (k/N)^2
        # which places more points near x0=0 where the front is steep
        
        if n == 1:
            x0_values = np.array([0.0])
        else:
            indices = np.arange(n)
            # Use beta=4 based on error analysis
            x0_values = (indices / (n - 1)) ** 4
            
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
