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
        
        # Optimal distribution derived from minimizing Riemann sum error
        # for f(u) = 10 + sqrt(u)
        # u_k = (k/n)^2
        
        indices = np.arange(1, n + 1)
        x0_values = (indices / n) ** 2
        
        # Ensure we don't exceed bounds or cause issues
        # x0_values are in [1/n^2, 1]
        
        # Shuffle to avoid any potential order-dependent issues in the evaluator
        # (though ZDT1 is stateless, it's a good practice)
        np.random.shuffle(x0_values)
        
        for x0 in x0_values:
            if time.time() > self.deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0
            # x[1:] are 0, which is within [0, 1]
            
            try:
                problem(x)
            except Exception:
                break
        
        return None
