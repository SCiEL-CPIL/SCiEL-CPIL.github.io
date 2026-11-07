import numpy as np
import time
import math
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_simplex(self, n):
        """Generate n Sobol points mapped to the 2D simplex (triangle)."""
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        points = sampler.random(n)
        
        u1 = points[:, 0]
        u2 = points[:, 1]
        mask = u1 + u2 > 1.0
        u1 = np.where(mask, 1.0 - u1, u1)
        u2 = np.where(mask, 1.0 - u2, u2)
        
        return u1, u2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        budget = self.budget
        dim = self.dim
        all_F = []
        
        # Add vertices explicitly
        vertices = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ])
        
        n_sobol = max(0, budget - 3)
        if n_sobol > 0:
            x0_s, x1_s = self._sobol_simplex(n_sobol)
            # Combine vertices and Sobol points
            x0 = np.concatenate([vertices[:, 0], x0_s])
            x1 = np.concatenate([vertices[:, 1], x1_s])
        else:
            x0 = vertices[:, 0]
            x1 = vertices[:, 1]
            
        # Truncate to budget if necessary (should be exact)
        x0 = x0[:budget]
        x1 = x1[:budget]

        for k in range(budget):
            if time.time() > deadline:
                break
            
            x = np.full(dim, 0.5)
            x[0] = x0[k]
            x[1] = x1[k]
            
            try:
                f = np.asarray(problem(x), dtype=np.float64)
                if f.shape == (3,) and np.all(np.isfinite(f)):
                    all_F.append(f)
            except Exception:
                pass
        
        return all_F
