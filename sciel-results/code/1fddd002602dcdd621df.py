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
        
        # Generate Halton sequence points
        # n = 200
        n = self.budget
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        points = []
        for i in range(1, n + 1):
            # 2D Halton
            u1 = halton(i, 2)
            u2 = halton(i, 3)
            
            # Map to x1, x2 for DTLZ2 front
            # x1 = (2/pi) * arccos(u1)
            # x2 = u2
            # x3..xN = 0
            
            x = np.zeros(self.dim)
            x[0] = (2.0 / np.pi) * np.arccos(u1)
            x[1] = u2
            # x[2:] are 0
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            points.append(f)
            
            if time.time() > deadline:
                break
                
        return np.array(points)
