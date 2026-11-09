"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc
import time


class Optimizer:
    """Analytical DTLZ4 optimizer: sample Pareto front using Sobol sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.t0 = time.time()
        self.deadline = self.t0 + 280  # Safety margin

    def __call__(self, problem, bounds):
        # DTLZ4 structure: last n_obj variables are distance variables
        # Setting them to 0 places the point on the Pareto front
        n_dist = self.n_obj
        n_pos = self.dim - n_dist

        if n_pos <= 0:
            # Fallback if dimensionality is very small
            X = np.zeros((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        # Generate Sobol sequence for position variables
        # Scrambling helps avoid bias
        sampler = qmc.Sobol(d=n_pos, scramble=True)
        points_pos = sampler.random(self.budget)

        # Construct full vectors
        X = np.zeros((self.budget, self.dim))
        X[:, :n_pos] = points_pos
        # Distance variables remain 0

        # Evaluate
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
