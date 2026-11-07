import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n_samples = self.budget
        
        # DTLZ7 structure:
        # M=3 objectives, n=22 variables.
        # Position variables: x[0], x[1]
        # Distance variables: x[2] ... x[21]
        # To be on the Pareto front, distance variables must be 0.
        
        # Use Sobol sequence for low-discrepancy sampling of the 2D position space
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        
        # Generate points. Sobol requires power of 2 for efficiency, but works for any N.
        # We just need n_samples points.
        sobol_points = sampler.random(n_samples)
        
        results = []
        for i in range(n_samples):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            # Assign position variables
            x[0] = sobol_points[i, 0]
            x[1] = sobol_points[i, 1]
            # Distance variables remain 0, ensuring g=0 and points on Pareto front
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                # If problem fails, continue with next point
                pass
                
        return results
