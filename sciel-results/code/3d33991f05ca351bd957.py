"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.stats as stats


class Optimizer:
    """Deterministic Sobol sequence sampling for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use Sobol sequence for low-discrepancy deterministic sampling
        # This provides better coverage of the decision space, especially for
        # multi-modal problems like WFG9, compared to random walks.
        sampler = stats.qmc.Sobol(d=self.dim, scramble=False)
        points = sampler.random(self.budget)
        
        # Map [0, 1] points to the given bounds
        X = lo + (hi - lo) * points
        
        # Evaluate all points
        # The problem function handles clipping and returns objectives
        for x in X:
            problem(x)
            
        return X
# EVOLVE-BLOCK-END
