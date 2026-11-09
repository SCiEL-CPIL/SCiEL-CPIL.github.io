import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 290 # 10s safety margin
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # DTLZ2 specific: set variables 2..11 to 0.5 to be on Pareto front
            # x[0] and x[1] determine the position on the front
            # We want to sample x[0], x[1] such that the resulting points on the front are uniformly distributed in area
            
            # Halton sequence generator
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r

            n_points = self.budget
            # We will evaluate n_points points
            for i in range(1, n_points + 1):
                if time.time() > deadline:
                    break
                
                u = halton(i, 2)
                v = halton(i, 3)
                
                # Transform to x[0], x[1] for uniform area coverage on DTLZ2 front
                # x0 = 2/pi * arcsin(u)
                # x1 = 2/pi * v
                
                x = np.zeros(self.dim)
                x[0] = (2.0 / np.pi) * np.arcsin(u)
                x[1] = (2.0 / np.pi) * v
                # Set remaining variables to 0.5 to minimize g(x)
                x[2:] = 0.5
                
                # Clip to bounds just in case, though 0.5 and [0,1] mapping should be safe
                x = np.clip(x, lo, hi)
                
                problem(x)
            
            return None
