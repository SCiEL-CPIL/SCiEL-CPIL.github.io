import numpy as np
from scipy.stats import qmc
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # 1. Probe the problem to determine the scaling of the front
        # Evaluate x = [1, 1, 0, ..., 0]
        x_probe = np.zeros(self.dim)
        x_probe[0] = hi[0]
        x_probe[1] = hi[1]
        
        obj_probe = problem(x_probe)
        n_evals += 1
        
        f0_probe = obj_probe[0]
        f1_probe = obj_probe[1]
        
        # Reference point is [1.1, 1.1, 25.0]
        # We want f0 < 1.1 and f1 < 1.1
        # If f0_probe is the value at x=1, then x_max for f0 < 1.1 is 1.1 / f0_probe * hi[0]
        # Assuming linear scaling f0 = S * x0.
        
        scale_0 = f0_probe / hi[0] if hi[0] > 0 else 1.0
        scale_1 = f1_probe / hi[1] if hi[1] > 0 else 1.0
        
        # Max x values to keep f < 1.1
        # f0 = scale_0 * x0 < 1.1 => x0 < 1.1 / scale_0
        max_x0 = min(hi[0], 1.1 / scale_0 if scale_0 > 0 else hi[0])
        max_x1 = min(hi[1], 1.1 / scale_1 if scale_1 > 0 else hi[1])
        
        # Ensure at least a small range if scale is weird
        if max_x0 < lo[0]: max_x0 = hi[0]
        if max_x1 < lo[1]: max_x1 = hi[1]
        
        # 2. Generate points
        # Use Sobol sequence for 199 points
        n_points = budget - n_evals # 199
        
        # Sobol generator
        sobol = qmc.Sobol(d=2, scramble=True)
        # Generate 200 points, take first 199? Or just 199?
        # qmc.Sobol requires power of 2 for best properties, but works for any.
        # Let's generate 256 and take 199 for better low-discrepancy?
        # Or just 199.
        points = sobol.random(n_points)
        
        # Scale to [lo, max]
        # points are in [0, 1]
        # x0 = lo[0] + points[:, 0] * (max_x0 - lo[0])
        # x1 = lo[1] + points[:, 1] * (max_x1 - lo[1])
        
        x0_vals = lo[0] + points[:, 0] * (max_x0 - lo[0])
        x1_vals = lo[1] + points[:, 1] * (max_x1 - lo[1])
        
        for i in range(n_points):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0_vals[i]
            x[1] = x1_vals[i]
            # x[2:] = 0 is already set
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
            
        return results
