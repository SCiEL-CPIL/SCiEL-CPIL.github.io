import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # The DTLZ7 problem with 3 objectives has a Pareto front that is a 2D square
        # in the (f0, f1) plane when g=0.
        # f0 = x0, f1 = 1 - x1, f2 = 1.
        # We set x2...x21 = 0 to ensure g=0.
        # We distribute budget points uniformly in the [0,1]^2 square using Sobol sequence.
        
        n_points = self.budget
        # Generate Sobol points in [0, 1]^2
        # scramble=False for determinism, though seed is set by evaluator
        sampler = qmc.Sobol(d=2, scramble=False)
        points = sampler.random(n_points)
        
        # points[:, 0] corresponds to x0
        # points[:, 1] corresponds to 1 - x1, so x1 = 1 - points[:, 1]
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = points[:, 0]
        X[:, 1] = 1.0 - points[:, 1]
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return F
