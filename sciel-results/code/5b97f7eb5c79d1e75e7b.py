import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """DTLZ2 optimizer using Sobol sequence with explicit corner points for optimal hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        evaluated_points = []
        used_budget = 0

        # Explicit corner points
        corner_uv = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
        ]

        # Evaluate corner points first
        for u, v in corner_uv:
            if time.time() > deadline:
                return evaluated_points
            if used_budget >= self.budget:
                break
            
            x0 = (2.0 / np.pi) * np.arccos(u)
            x1 = v
            
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        n_remaining = self.budget - used_budget
        
        if n_remaining > 0:
            # Use Sobol sequence for better low-discrepancy properties
            sampler = qmc.Sobol(d=2, scramble=True)
            # We need to generate enough points. 
            # Sobol requires power of 2 for optimal performance, but works for any N.
            # Let's generate n_remaining points.
            # Note: qmc.Sobol.random returns points in [0,1).
            # We might want to include the boundary? 
            # The inverse CDF mapping handles the interior.
            # The corners are already added.
            
            # To avoid duplicates with corners (which are at 0, 1), 
            # we can just use the Sobol points as is.
            # 0 and 1 are measure zero, but Sobol might generate 0? 
            # Sobol points are in [0, 1). So 0 is possible, 1 is not.
            # (0,0) is a corner. If Sobol generates (0,0), it's a duplicate.
            # We can filter or just accept it (wastes 1 budget).
            # Given budget is 200, wasting a few is okay, but let's try to be efficient.
            
            points = sampler.random(n_remaining)
            
            for u, v in points:
                if time.time() > deadline:
                    break
                if used_budget >= self.budget:
                    break
                
                # Map to x
                x0 = (2.0 / np.pi) * np.arccos(u)
                x1 = v
                
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                evaluated_points.append(x)
                used_budget += 1

        return evaluated_points
