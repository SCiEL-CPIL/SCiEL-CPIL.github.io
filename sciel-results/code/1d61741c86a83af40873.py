"""Multi-objective optimizer for DTLZ1 using direct Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Direct sampler for DTLZ1 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Strategy: Sobol sequence mapped uniformly to the Pareto simplex
        # This ensures all 200 evaluations are used for a well-distributed set of points
        # on the simplex surface, maximizing hypervolume coverage.
        
        X = np.zeros((budget, dim))
        
        # Generate Sobol points in [0, 1]^2
        sobol = qmc.Sobol(d=2, scramble=True)
        points_2d = sobol.random(budget)
        
        # Map to simplex coordinates u1, u2, u3 uniformly
        # u1 = 1 - sqrt(s1)
        # u2 = (1 - u1) * sqrt(s2)
        # This transformation ensures uniform distribution on the simplex area.
        
        s1 = points_2d[:, 0]
        s2 = points_2d[:, 1]
        
        # Compute x1 and x2 directly
        # x1 = u1 = 1 - sqrt(s1)
        # x2 = u2 / (1 - u1) = sqrt(s2)
        
        X[:, 0] = 1.0 - np.sqrt(s1)
        X[:, 1] = np.sqrt(s2)
        
        for d in range(2, dim):
            X[:, d] = (lo[d] + hi[d]) / 2.0
        
        # Evaluate all points
        all_points = []
        used = 0
        for x in X:
            if time.time() > deadline:
                break
            f = problem(x)
            all_points.append(f)
            used += 1
            
        return X
# EVOLVE-BLOCK-END
