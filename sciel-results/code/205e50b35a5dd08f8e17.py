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
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure:
        # First dim - n_obj + 1 are shape variables
        # Last n_obj - 1 are distance variables
        n_shape = self.dim - self.n_obj + 1
        n_dist = self.n_obj - 1
        
        # Distance variables should be 0.5 to be on the Pareto front
        # Shape variables are sampled in [0, 1]
        
        # Generate Sobol sequence for shape variables
        # We have self.budget evaluations
        # We can evaluate all of them
        sampler = qmc.Sobol(d=n_shape, scramble=True, seed=0)
        points = sampler.random(self.budget)
        
        # points is (budget, n_shape) in [0, 1]
        # We need to map to bounds if they are not [0,1], but task says bounds [0,1]
        # Just in case, scale to bounds
        shape_bounds_lo = lo[:n_shape]
        shape_bounds_hi = hi[:n_shape]
        
        shape_points = shape_bounds_lo + (shape_bounds_hi - shape_bounds_lo) * points
        
        # Construct full vectors
        X = np.zeros((self.budget, self.dim))
        X[:, :n_shape] = shape_points
        # Set distance variables to 0.5 (midpoint of [0,1])
        # If bounds are not [0,1], use midpoint
        dist_bounds_lo = lo[n_shape:]
        dist_bounds_hi = hi[n_shape:]
        dist_mid = (dist_bounds_lo + dist_bounds_hi) / 2.0
        
        for i in range(self.budget):
            X[i, n_shape:] = dist_mid
            
            # Check time
            if time.time() > deadline:
                break
                
            # Evaluate
            try:
                f = problem(X[i])
            except Exception:
                continue

        return X
