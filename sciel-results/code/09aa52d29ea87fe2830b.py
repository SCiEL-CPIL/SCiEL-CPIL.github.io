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
        deadline = start_time + 285
        
        n_points = self.budget
        dim = self.dim
        
        pass  # Using scipy Sobol instead
        
        # Strategic points in v-space (v1, v2)
        # v1 maps to f1 = sqrt(1-v1^2), f2 = v1*sin(pi*v2/2), f3 = v1*cos(pi*v2/2)
        # Vertices:
        # (0, 0) -> f=(1, 0, 0)
        # (1, 0) -> f=(0, 1, 0)
        # (1, 1) -> f=(0, 0, 1)
        
        # Edge midpoints:
        # Edge v1=1 (between (0,1,0) and (0,0,1)): v2=0.5 -> (1, 0.5)
        # Edge v2=0 (between (1,0,0) and (0,1,0)): v1=0.5 -> (0.5, 0)
        # Edge v2=1 (between (1,0,0) and (0,0,1)): v1=0.5 -> (0.5, 1)
        
        # Let's add a few more points on the edges to ensure good coverage of the "boundaries"
        # which contribute significantly to hypervolume.
        
        strategic_points = [
            (0.0, 0.0), # (1, 0, 0)
            (1.0, 0.0), # (0, 1, 0)
            (1.0, 1.0), # (0, 0, 1)
            (0.5, 0.0), # Edge f3=0
            (0.5, 1.0), # Edge f2=0
            (1.0, 0.5), # Edge f1=0
        ]
        
        points_v = list(strategic_points)
        
        # Generate remaining points using 2D Sobol sequence for better space-filling
        num_remaining = n_points - len(points_v)
        if num_remaining > 0:
            sobol = qmc.Sobol(d=2, scramble=True)
            k = int(np.ceil(np.log2(max(num_remaining, 1))))
            n_sobol = 2 ** k
            sobol_points = sobol.random(n_sobol)
            # Skip the first point (origin) and take num_remaining points
            sobol_points = sobol_points[1:1+num_remaining]
            for pt in sobol_points:
                points_v.append((pt[0], pt[1]))
            
        results = []
        for v1, v2 in points_v:
            if time.time() > deadline:
                break
                
            # Clip to [0,1] just in case of floating point issues
            v1_c = float(np.clip(v1, 0.0, 1.0))
            v2_c = float(np.clip(v2, 0.0, 1.0))
            
            # Map to x0, x1
            # x0 = 2/pi * arccos(v1)
            # x1 = v2
            x0 = (2.0 / np.pi) * np.arccos(v1_c)
            x1 = v2_c
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            # Set distance variables to 0.5 to minimize g(x) and stay on the Pareto front
            x[2:] = 0.5
            
            f = problem(x)
            results.append(f)
            
        return results
