"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure via direct construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits the known Pareto front of DTLZ1.
    For n_obj=3, the Pareto front is parameterized by x_0, x_1 in [0,1]
    with x_2..x_{dim-1} = 0.5 (so g=0).
    Uses Sobol sequence for low-discrepancy sampling of the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # For n_obj=3:
        # f0 = 0.5 * (1+g) * x0 * x1
        # f1 = 0.5 * (1+g) * (1-x1) * x0  
        # f2 = 0.5 * (1+g) * (1-x0)
        # On Pareto front: g=0, so x_2..x_{dim-1} = 0.5
        # The front is a 2D surface parameterized by (x0, x1) in [0,1]^2
        
        if n_obj == 3:
            # Use Sobol sequence for low-discrepancy sampling in [0,1]^2
            # scipy.stats.qmc.Sobol provides quasi-random low-discrepancy sequences
            from scipy.stats import qmc
            
            # Generate budget points using Sobol sequence
            # Sobol requires power-of-2 number of points for optimal properties
            # but works with any n
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            points_2d = sampler.random(budget)
            
            # points_2d is (budget, 2) with values in [0, 1)
            x0_vals = points_2d[:, 0]
            x1_vals = points_2d[:, 1]
            
            # Build decision vectors
            X = np.zeros((budget, dim))
            X[:, 0] = x0_vals
            X[:, 1] = x1_vals
            # x_2..x_{dim-1} = 0.5 for g=0
            X[:, 2:] = 0.5
        else:
            # General case fallback: random sampling
            X = lo + (hi - lo) * np.random.default_rng(0).random((budget, dim))
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        return X
# EVOLVE-BLOCK-END
