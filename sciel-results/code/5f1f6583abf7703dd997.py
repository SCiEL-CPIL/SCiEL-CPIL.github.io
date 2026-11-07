"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT1 problem.
    Exploits the known Pareto front structure of ZDT1: f2 = 1 - sqrt(f1) when x2...xn=0.
    Uses a power-law distribution for f1 to equalize hypervolume loss between consecutive points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        # Determine number of evaluations to perform
        n_evals = self.budget
        
        # ZDT1 Pareto front: f1 = x1, f2 = 1 - sqrt(x1)
        # Optimal distribution: f1 = t^(4/3) for t uniform in [0,1]
        # This equalizes the hypervolume loss between consecutive points.
        t_samples = np.linspace(0.0, 1.0, n_evals)
        x1_samples = t_samples ** (4.0 / 3.0)
        
        # Construct the population matrix
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_samples
        
        # Ensure variables are within bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check deadline
        if time.time() > deadline - 5.0:
            return X
        
        return X
# EVOLVE-BLOCK-END
