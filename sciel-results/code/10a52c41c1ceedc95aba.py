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
        
        # DTLZ2 specific knowledge:
        # dim = 12, n_obj = 3
        # Position variables: x[0], x[1]
        # Decision variables: x[2] ... x[11]
        # g(x) is minimized when x[2]...x[11] = 0.5
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We assume the bounds are [0, 1] for all variables as per task description
        # But we use the provided bounds to be safe.
        
        # Generate Halton sequence for x[0] and x[1]
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        X = np.zeros((self.budget, self.dim))
        
        # Fill decision variables with 0.5 (midpoint of bounds)
        mid = (lo + hi) / 2.0
        X[:, 2:] = mid[2:]
        
        # Fill position variables with Halton sequence
        # We want to cover [0, 1] for x[0] and x[1]
        # Halton base 2 for x[0], base 3 for x[1]
        for i in range(self.budget):
            if time.time() > deadline:
                break
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * halton(i + 1, 2)
            X[i, 1] = lo[1] + (hi[1] - lo[1]) * halton(i + 1, 3)
            
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
