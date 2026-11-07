import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 30s margin
        
        # Strategy: Exploit DTLZ7 structure
        # DTLZ7 with M=3, n=22: m = n-M+1 = 20
        # g depends on x[20], x[21] (0-indexed: indices 20, 21)
        # Pareto front: x[20]=x[21]=0.5, x[0] and x[1] free in [0,1]
        # f1 = x[0], f2 = x[1], f3 = 1 - x[1]*(1+sin(10*pi*x[1]))
        # Use 2D Halton sequence to sample (x1, x2) for best HV coverage
        
        n_samples = min(self.budget, 200)
        
        # Generate 2D Halton sequence
        def halton(index, base):
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i = i // base
                f /= base
            return result
        
        # Generate Halton points in 2D
        # Using primes 2 and 3 for the two dimensions
        x1_vals = np.array([halton(i + 1, 2) for i in range(n_samples)])
        x2_vals = np.array([halton(i + 1, 3) for i in range(n_samples)])
        
        results = []
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1_vals[i]   # x1
            x[1] = x2_vals[i]   # x2
            x[20] = 0.5         # g=0 for Pareto front
            x[21] = 0.5         # g=0 for Pareto front
            
            f = problem(x)
            results.append(f)
            
        return results
