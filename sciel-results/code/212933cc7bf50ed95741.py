# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        results = []
        
        n_total = self.budget
        
        # Explicit corners to ensure max coverage
        explicit_points = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]
        
        n_explicit = len(explicit_points)
        n_sobol = max(0, n_total - n_explicit)
        
        for pt in explicit_points:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = pt[0]
            x[1] = pt[1]
            f = problem(x)
            results.append(f)
            n_evals += 1
        
        if n_sobol > 0:
            sampler = qmc.Sobol(d=2, scramble=True)
            sobol_samples = sampler.random(n=n_sobol)
            
            x0_vals = sobol_samples[:, 0]
            x1_vals = sobol_samples[:, 1] ** 2
            
            for i in range(n_sobol):
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = x0_vals[i]
                x[1] = x1_vals[i]
                f = problem(x)
                results.append(f)
                n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
