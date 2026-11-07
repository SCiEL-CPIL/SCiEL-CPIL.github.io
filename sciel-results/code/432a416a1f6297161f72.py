"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Deterministic Global Optimization using Sobol sequence for full budget coverage."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initialize Sobol sampler with scrambling for better statistical properties
        # Seed is fixed for reproducibility, but scrambling makes it quasi-random
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        
        # Generate the full set of points in [0, 1)^dim
        points_unit = sampler.random(budget)
        
        # Map to the search space [-5, 5]^dim
        # Linear transformation: x = (b - a) * y + a
        # Here a = -5, b = 5, so x = 10 * y - 5
        points_search = points_unit * 10.0 - 5.0

        best_x = None
        best_f = np.inf

        # Evaluate all points sequentially
        # This ensures that the first few points provide immediate coverage
        # and the sequence property ensures error reduction is monotonic-ish
        for i in range(budget):
            x = points_search[i]
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
