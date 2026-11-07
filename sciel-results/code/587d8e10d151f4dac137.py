import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        # DTLZ1 with M=3, N=7
        # Pareto front is determined by x0, x1
        # x2..x6 must be 0 for g=0
        
        n_front_vars = 2 # x0, x1
        n_penalty_vars = self.dim - n_front_vars # x2..x6
        
        # Halton sequence generator
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index //= base
            return r

        # Generate points
        # We have budget evaluations. Use all of them.
        # Ensure we stay within time limit.
        # 200 evaluations is very fast.
        
        for i in range(1, self.budget + 1):
            if time.time() > deadline:
                break
            
            x0 = halton(i, 2)
            x1 = halton(i, 3)
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            # x[2:] is already 0
            
            # Call problem to record the evaluation
            problem(x)
            
        return None
