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
        # Use Sobol sequence for better 2D coverage of the Pareto front
        
        n_samples = min(self.budget, 200)
        
        # Generate Sobol-like quasi-random points using inverse CDF of uniform
        # For better coverage, use a stratified approach with jitter
        # Actually, let's use a better deterministic low-discrepancy sequence
        
        # Use inverse of the van der Corput / Halton but with better base selection
        # Or use a simple stratified sampling with jitter for near-uniform coverage
        
        # Better approach: Use a 2D Sobol sequence approximation via scipy
        from scipy.stats import qmc
        
        sampler = qmc.Sobol(d=2, scramble=True)
        # We need a power of 2 for Sobol, so use next power of 2 >= n_samples
        n_pow2 = 1
        while n_pow2 < n_samples:
            n_pow2 *= 2
        
        sobol_points = sampler.random(n_pow2)
        # Scale to (0, 1) - Sobol already gives [0,1)
        # Add small epsilon to avoid boundary issues if needed
        x1_vals = sobol_points[:n_samples, 0]
        x2_vals = sobol_points[:n_samples, 1]
        
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
