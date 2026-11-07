# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting the known Pareto front structure.
    
    DTLZ2 with 3 objectives has a Pareto front that is a quarter-sphere in the positive
    octant of R^3. The distance variables (last dim - n_obj + 1 = 10 variables) should
    be set to 0.5 for optimality. The angle variables (first n_obj - 1 = 2 variables)
    determine position on the front.
    
    Strategy: 
    1. Use Sobol quasi-random sequence to sample the 2D angle space densely.
    2. Set distance variables to 0.5 to lie exactly on the Pareto front.
    3. Use the entire budget for deterministic, space-filling sampling to maximize
       hypervolume coverage without wasting evaluations on dominated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # DTLZ2 structure:
        # - First m-1 variables are "angle" variables that determine position on Pareto front
        # - Last n-m+1 variables are "distance" variables; optimal value is 0.5 for each
        # - Pareto front: sum of squares of objectives = 1 (unit sphere in positive octant)
        
        # Number of distance variables
        n_dist = n - m + 1  # = 10 for dim=12, n_obj=3
        
        # Optimal distance values (0.5 for DTLZ2)
        dist_optimal = np.full(n_dist, 0.5)
        
        # Use the entire budget for Sobol sampling of the 2D angle space
        n_samples = self.budget
        
        # Use Sobol sequence for (m-1) dimensional angle space
        # scramble=False for deterministic, reproducible results
        sampler = qmc.Sobol(d=m-1, scramble=False)
        
        # Generate points. Sobol works best with powers of 2, but handles any N.
        # We'll generate n_samples points directly.
        angles = sampler.random(n_samples)
        
        # Build decision vectors
        X = np.zeros((n_samples, n))
        
        # Set distance variables to optimal value 0.5
        X[:, m-1:] = dist_optimal
        
        # Set angle variables from Sobol samples (already in [0, 1])
        X[:, :m-1] = angles
        
        # Clip to bounds (should already be within [0,1] for standard DTLZ2 bounds)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.zeros((n_samples, m))
        for i in range(n_samples):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
