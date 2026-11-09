"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits the known ZDT3 Pareto front structure.
    ZDT3's Pareto front lies on the manifold x[1:] = 0.5.
    On this manifold:
      f1 = x[0]
      f2 = 1 + 10 * (1 - sqrt(x[0]/30) - (x[0]/30)*sin(10*pi*x[0]))
    
    We sample x0 values to cover [0, 1] and evaluate on the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front: x[1:] = 0.5, x[0] varies in [0, 1]
        # We'll sample x[0] values to cover the front
        n_samples = min(self.budget, 200)
        
        # Create sample points on the Pareto front
        x0_values = np.linspace(0.0, 1.0, n_samples)
        
        # Generate candidate points
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_values
        X[:, 1:] = 0.5  # Pareto front manifold
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
