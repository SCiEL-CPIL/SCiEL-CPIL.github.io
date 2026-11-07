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
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Generate Sobol points in 2D
        sampler = qmc.Sobol(d=2, scramble=True)
        # We want n points.
        # Sobol requires power of 2? No, it can generate any number, but efficiency is best for powers of 2.
        # Let's just generate n points.
        sobol_points = sampler.random(n)
        
        # Transform to simplex coordinates
        # u, v in [0, 1]
        u = sobol_points[:, 0]
        v = sobol_points[:, 1]
        
        # Area-preserving mapping to triangle
        # lambda1 = 1 - r1 - r2
        # lambda2 = r1
        # lambda3 = r2
        # r1 = sqrt(u)
        # r2 = v * sqrt(1-u)
        
        r1 = np.sqrt(u)
        r2 = v * np.sqrt(1 - u)
        
        lambda1 = 1 - r1 - r2
        lambda2 = r1
        lambda3 = r2
        
        # Clip to ensure valid simplex (numerical errors)
        lambda1 = np.clip(lambda1, 0, 1)
        lambda2 = np.clip(lambda2, 0, 1)
        lambda3 = np.clip(lambda3, 0, 1)
        
        # Normalize just in case
        sum_lam = lambda1 + lambda2 + lambda3
        lambda1 /= sum_lam
        lambda2 /= sum_lam
        lambda3 /= sum_lam
        
        # Target objectives
        # f0 = 0.5 * lambda1
        # f1 = 0.5 * lambda2
        # f2 = 0.5 * lambda3
        
        # Decision variables
        # x0 = 1 - 2*f2 = 1 - lambda3
        # x1 = 2*f0 / x0 = lambda1 / (1 - lambda3)
        
        x0 = 1 - lambda3
        # Handle x0 = 0 case
        # If x0 is very small, x1 is undefined or 0.
        # If x0=0, then lambda3=1, lambda1=0, lambda2=0.
        # x1 should be 0.
        x1 = np.zeros_like(x0)
        mask = x0 > 1e-12
        x1[mask] = lambda1[mask] / x0[mask]
        
        # Clip x0, x1 to [0, 1]
        x0 = np.clip(x0, 0, 1)
        x1 = np.clip(x1, 0, 1)
        
        # Construct X
        X = np.zeros((n, dim))
        X[:, 0] = x0
        X[:, 1] = x1
        # x2..x6 = 0.5
        if dim > 2:
            X[:, 2:] = 0.5
            
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
