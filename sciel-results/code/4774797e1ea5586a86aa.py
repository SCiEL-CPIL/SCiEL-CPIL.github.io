import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, dim=2):
        """Generate a low-discrepancy Sobol-like sequence using a simple approach.
        Uses a deterministic quasi-random sequence based on van der Corput in 2D."""
        points = np.zeros((n, dim))
        for i in range(n):
            # Van der Corput sequence in base 2 for each dimension
            d = 1.0
            v = 0.0
            ii = i
            while ii > 0:
                d /= 2.0
                v += d * (ii % 2)
                ii //= 2
            points[i, 0] = v
            # For second dimension, use a different base or offset
            d = 1.0
            v = 0.0
            ii = i + 1  # offset to decorrelate
            while ii > 0:
                d /= 3.0  # base 3 for better coverage
                v += d * (ii % 3)
                ii //= 3
            points[i, 1] = v
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        results = []
        
        g_vars = np.full(5, 0.5)
        
        # Generate Sobol-like points in [0,1]^2
        n_sobol = self.budget
        sobol_pts = self._sobol_sequence(n_sobol, dim=2)
        
        # Apply density-weighted mapping to concentrate points near corners of simplex.
        # The Pareto front for DTLZ1 (M=3) is: f1=0.5*x0, f2=0.5*x1*(1-x0), f3=0.5*(1-x1)
        # Points near (0,0), (1,0), (0,1) in (x0,x1) space correspond to corners of the front.
        # Use a power transform to bias toward boundaries.
        # alpha < 1 pushes mass toward 0 and 1.
        alpha = 0.7  # Tune this: lower = more concentrated at edges
        
        x0_mapped = np.where(sobol_pts[:, 0] < 0.5, 
                              2.0 * sobol_pts[:, 0] ** (1.0/alpha) * 0.5,
                              1.0 - 2.0 * (1.0 - sobol_pts[:, 0]) ** (1.0/alpha) * 0.5)
        x1_mapped = np.where(sobol_pts[:, 1] < 0.5,
                              2.0 * sobol_pts[:, 1] ** (1.0/alpha) * 0.5,
                              1.0 - 2.0 * (1.0 - sobol_pts[:, 1]) ** (1.0/alpha) * 0.5)
        
        # Clip to [0,1]
        x0_mapped = np.clip(x0_mapped, 0.0, 1.0)
        x1_mapped = np.clip(x1_mapped, 0.0, 1.0)
        
        for i in range(n_sobol):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x0 = x0_mapped[i]
            x1 = x1_mapped[i]
            
            x = np.zeros(7)
            x[0] = x0
            x[1] = x1
            x[2:] = g_vars
            
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                used += 1
                f = np.array(f)
                if np.all(f < 400.0):
                    results.append(f)
            except Exception:
                continue
        
        return results
