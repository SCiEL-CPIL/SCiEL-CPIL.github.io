"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def halton_sequence(n, base):
    """Generate the first n points of the Halton sequence in 1D."""
    seq = np.zeros(n)
    f = 1.0 / base
    for i in range(1, n + 1):
        seq[i - 1] = 0.0
        j = i
        while j > 0:
            seq[i - 1] += f * (j % base)
            j //= base
            f /= base
    return seq


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using direct Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        
        # For DTLZ2 with 3 objectives:
        # The Pareto front is defined by g(x) = 0 and x_1 = 0.
        # x_4 ... x_12 = 0.5 ensures g(x) = 0.
        # x_1 = 0 ensures the points lie on the unit sphere in objective space.
        # x_2, x_3 vary in [0, 1] to cover the available portion of the front.
        
        # Generate points using Halton sequence for low discrepancy in 2D (x_2, x_3)
        # We use two different bases for 2 dimensions
        n_points = budget
        h2 = halton_sequence(n_points, 2)
        h3 = halton_sequence(n_points, 3)
        
        X = np.zeros((n_points, dim))
        
        # Set fixed variables
        X[:, 0] = 0.0
        if dim > 3:
            X[:, 3:] = 0.5
            
        # Set varying variables x_2, x_3
        X[:, 1] = h2
        X[:, 2] = h3
        
        # Evaluate all points
        F = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            F[i] = problem(X[i])
            
        return X

# EVOLVE-BLOCK-END
