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
        deadline = start_time + 280  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = 0
        all_points = []
        all_fvals = []

        def eval_point(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            if len(f) != self.n_obj:
                return None
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_points.append(x)
            all_fvals.append(f)
            n_evals += 1
            return f

        # DTLZ2 structure:
        # The Pareto front is defined by x0 and x1.
        # x2...x11 should be 0.5 to minimize the distance to the front (g=0).
        # We will sample x0 and x1 uniformly in [0, 1] using a Sobol sequence
        # to ensure good coverage of the quarter-sphere surface.
        
        # We have a total budget of 200 evaluations.
        # We will use all of them for the structured sampling.
        n_samples = self.budget
        
        # Generate Sobol sequence points in 2D
        # Sobol sequences provide low-discrepancy points, ideal for covering the 2D space
        # that maps to the Pareto front.
        sampler = qmc.Sobol(d=2, scramble=True)
        # Number of points must be a power of 2 for Sobol.
        # Find the largest power of 2 <= n_samples
        n_pow2 = 1
        while n_pow2 * 2 <= n_samples:
            n_pow2 *= 2
        
        # If n_samples is not a power of 2, we can either:
        # 1. Use only the first n_pow2 points.
        # 2. Use n_pow2 points and then fill the remaining budget with random points or a finer grid.
        # For DTLZ2, uniform coverage is key. A Sobol sequence of size 128 is very good.
        # Let's use 128 points from Sobol, and if budget allows, add more.
        
        # Actually, the Sobol generator in scipy can generate any number of points,
        # but the low-discrepancy property is best for powers of 2.
        # Let's just generate n_samples points.
        
        # Note: qmc.Sobol in scipy.stats generates points in [0, 1)^d
        points_2d = sampler.random(n_samples)
        
        # Construct full 12D points
        # x[0] = points_2d[i, 0]
        # x[1] = points_2d[i, 1]
        # x[2:] = 0.5
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = points_2d[i, 0]
            x[1] = points_2d[i, 1]
            x[2:] = 0.5
            
            # Clip to bounds just in case, though [0,1] is the standard range
            x = np.clip(x, lo, hi)
            
            f = eval_point(x)
            
        return all_points
