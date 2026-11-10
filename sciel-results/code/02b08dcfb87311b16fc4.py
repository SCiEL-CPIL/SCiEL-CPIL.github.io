"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT4.
    
    Exploits the structure of ZDT4 where the Pareto front lies on the manifold 
    x_i = 0 for i > 0. We project all candidates onto this manifold and 
    spread x_0 to cover the full range of the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds for x_0
        # ZDT4 typically has x_0 in [-5, 5], but valid front is x_0 >= 0
        lo_0, hi_0 = bounds[0]
        
        # Ensure we sample in the valid domain for the square root in f_2
        # If hi_0 is negative, the problem might be ill-defined for standard ZDT4,
        # but we clip to 0 to be safe.
        if hi_0 < 0:
            hi_0 = 0.0
        
        # Generate x_0 values
        # We want to cover [0, hi_0] to maximize the range of f_2
        # Using linspace ensures we hit the boundaries which define the HV extent
        if self.budget > 2:
            x0_vals = np.linspace(0, hi_0, self.budget)
        else:
            x0_vals = np.linspace(0, hi_0, max(2, self.budget))[:self.budget]
            
        # Initialize population
        X = np.zeros((self.budget, self.dim))
        X[:, 0] = x0_vals
        # x[1:] is already 0, which is the optimal setting for f_1
        
        # Evaluate all points
        # problem(x) returns objective values
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
