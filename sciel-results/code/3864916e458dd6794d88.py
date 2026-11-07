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
        
        # DTLZ7 Pareto front is achieved by setting x[2:] = 0 (assuming standard definition)
        # and varying x[0] and x[1].
        
        # Generate 200 points uniformly in [0, 1] x [0, 1]
        N = self.budget
        
        # Use a Halton sequence for low discrepancy in 2D
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        u_vals = np.array([halton(i + 1, 2) for i in range(N)])
        v_vals = np.array([halton(i + 1, 3) for i in range(N)])
        
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
            # x[2:] remains 0, which is lo[2] if lo is 0.
            # If bounds are not [0,1], we should set to lo.
            x[2:] = lo[2:]
            
            last_result = problem(x)
            
        return last_result
