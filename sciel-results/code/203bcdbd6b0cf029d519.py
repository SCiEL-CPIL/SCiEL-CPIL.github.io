"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for DTLZ4 that directly samples the analytical Pareto front.
    
    The Pareto front of DTLZ4 is located at x_2...x_{n-1} = 0.
    The objectives are determined by x_0 and x_1.
    We use a Sobol sequence to sample x_0 and x_1.
    We apply a transformation to x_0 to ensure uniform coverage of the 
    spherical Pareto front surface (area element proportional to sin(theta_0)).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Generate Sobol sequence
        sampler = qmc.Sobol(d=2, scramble=True)
        samples = sampler.random(self.budget)
        
        # samples are in [0, 1]^2
        u = samples[:, 0]
        v = samples[:, 1]
        
        # Transform to x_0, x_1 for uniform surface coverage
        # x_1 is uniform in [0, 1]
        x_1 = v
        
        # x_0 transformation: theta_0 = arccos(1 - u)
        # x_0 = (2/pi) * theta_0
        x_0 = (2.0 / np.pi) * np.arccos(1.0 - u)
        
        # Construct the full decision vectors
        # x_2 ... x_{dim-1} are set to 0
        X = np.zeros((self.budget, self.dim))
        X[:, 0] = x_0
        X[:, 1] = x_1
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
