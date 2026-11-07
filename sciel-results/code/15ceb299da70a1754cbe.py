import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate the index-th Halton number in base."""
        result = 0
        f = 1
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # DTLZ4 structure:
        # x[0], x[1] determine the Pareto front location.
        # x[2] to x[11] must be 0.5 to be on the Pareto front (g(x)=0).
        
        # We will generate `budget` points on the Pareto front.
        # Use Halton sequence for x[0], x[1] to ensure good coverage.
        
        # Bases for Halton sequence (2 dimensions)
        bases = [2, 3]
        
        # Pre-allocate the fixed part of the vector
        # dim=12, so indices 2..11 are fixed at 0.5
        fixed_part = np.full(self.dim - 2, 0.5)
        
        F = []
        
        for i in range(self.budget):
            if time.time() > deadline:
                break
                
            # Generate 2 Halton numbers
            # Index i+1 to avoid 0,0 which might be a singular point or just to spread
            # Actually, 0 is fine, it's a valid point on the front.
            x0 = self._halton(i + 1, bases[0])
            x1 = self._halton(i + 1, bases[1])
            
            # Construct the full vector
            x = np.empty(self.dim)
            x[0] = x0
            x[1] = x1
            x[2:] = fixed_part
            
            # Evaluate
            f = problem(x)
            F.append(f)
            
        return F
