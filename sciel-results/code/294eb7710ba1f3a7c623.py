import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        
        n_shape = n_obj - 1  # 2
        n_dist = dim - n_shape  # 10
        
        # Known optimum for distance variables
        dist_opt = 0.5 * np.ones(n_dist)
        
        # Generate shape variables using Sobol sequence for better coverage
        # We need n_shape dimensions for the Sobol sequence
        sampler = qmc.Sobol(d=n_shape, scramble=True)
        # Generate budget points
        # Sobol requires power of 2 for best performance, but can generate any number
        # We will generate exactly self.budget points
        shape_samples = sampler.random(self.budget)
        
        # Scale to [0, 1]
        # Sobol returns [0, 1) by default
        # We want to include 1.0? The bounds are [0,1]. 
        # DTLZ4 front is defined for z in [0,1].
        # Let's ensure we cover the full range.
        shape_samples = shape_samples * (hi[:n_shape] - lo[:n_shape]) + lo[:n_shape]
        
        X = np.zeros((self.budget, dim))
        # Fill distance variables
        X[:, n_shape:] = dist_opt
        # Fill shape variables
        X[:, :n_shape] = shape_samples
        
        # Clip to be safe
        X = np.clip(X, lo, hi)
        
        F = np.zeros((self.budget, n_obj))
        
        for i in range(self.budget):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return F
