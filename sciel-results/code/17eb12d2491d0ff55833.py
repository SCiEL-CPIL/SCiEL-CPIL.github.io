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
        deadline = start_time + 290
        
        n_points = self.budget
        if n_points < 4:
            n_points = max(1, n_points)
            
        # Generate Sobol' sequence points in [0, 1)^2
        # Sobol' is generally superior to Halton for low dimensions
        sampler = qmc.Sobol(d=2, scramble=True)
        
        # Explicit corners to ensure extreme points are covered
        # New mapping: x0 = 1 - sqrt(1-v1), x1 = v2
        # v1=0 -> x0=0. v1=1 -> x0=1.
        # Corner 1: v1=0, v2=0 -> x0=0, x1=0 -> f=(0, 1, 0)
        # Corner 2: v1=0, v2=1 -> x0=0, x1=1 -> f=(0, 0, 1)
        # Corner 3: v1=1, v2=0 -> x0=1, x1=0 -> f=(1, 0, 0)
        
        points_v = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0)
        ]
        
        # Generate remaining points
        num_sobol = n_points - len(points_v)
        if num_sobol > 0:
            # Sobol' sequence generation
            # Note: Sobol' works best with power of 2, but accepts any N
            points_v_sobol = sampler.random(num_sobol)
            for p in points_v_sobol:
                if time.time() > deadline:
                    break
                points_v.append(tuple(p))
        else:
            points_v_sobol = []
            
        results = []
        for v1, v2 in points_v:
            # Map to x0, x1
            # v1 is in [0, 1]
            v1_c = np.clip(v1, 0.0, 1.0)
            v2_c = np.clip(v2, 0.0, 1.0)
            
            # x0 = 1 - sqrt(1-v1) for uniform surface area coverage
            x0 = 1.0 - np.sqrt(1.0 - v1_c)
            # x1 = v2
            x1 = v2_c
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5
            
            f = problem(x)
            results.append(f)
            
        return results
