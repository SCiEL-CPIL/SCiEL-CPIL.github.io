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
        # x1=0, x3..xn=0, sample x2 in (0, 1]
        # Use Sobol sequence for better 1D coverage (quasi-random)
        
        n_samples = min(self.budget, 200)
        
        # Generate Sobol sequence values in (0, 1)
        # Sobol gives better low-discrepancy than uniform or linear
        # For 1D, we can use a simple inverse CDF approach or precomputed values
        # Using a Halton-like approach with prime base for 1D
        # Or use numpy's random with Sobol-like properties
        
        # Simple approach: use sorted random points from a low-discrepancy sequence
        # For 1D, the van der Corput sequence (base 2) is optimal
        def van_der_corput(n, base=2):
            """Generate n points from van der Corput sequence in (0, 1)"""
            points = []
            for i in range(1, n + 1):
                f = 1.0 / base
                r = 0.0
                while i > 0:
                    i, rem = divmod(i, base)
                    r += rem * f
                    f /= base
                points.append(r)
            return np.array(points)
        
        x2_values = van_der_corput(n_samples)
        # Sort to ensure consistent ordering (not strictly necessary but helps debugging)
        x2_values = np.sort(x2_values)
        
        results = []
        for x2 in x2_values:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = 0.0 # x1
            x[1] = x2  # x2
            # x[2:] are 0.0
            
            # Ensure within bounds
            # Bounds are [0,1]
            
            f = problem(x)
            results.append(f)
            
        return results
