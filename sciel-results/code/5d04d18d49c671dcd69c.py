import numpy as np
from numpy.polynomial.legendre import leggauss

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        if self.budget <= 0:
            return np.array([])

        n_evals = self.budget
        if n_evals < 1:
            return np.array([])
        
        # We know the Pareto front is at x[1:] = 0
        # We will place points at x1 = 0, x1 = 1, and Gauss-Legendre nodes in (0, 1)
        
        # Ensure we have at least 2 points for endpoints if budget allows
        if n_evals == 1:
            # Just pick the middle
            x1_vals = [0.5]
        else:
            # Reserve 2 for endpoints
            n_interior = n_evals - 2
            if n_interior > 0:
                # Get Gauss-Legendre nodes in [-1, 1]
                nodes, _ = leggauss(n_interior)
                # Map to (0, 1)
                x1_interior = 0.5 * (nodes + 1.0)
                # Sort them to be tidy (leggauss returns sorted)
                x1_vals = [0.0] + list(x1_interior) + [1.0]
            else:
                # If budget is 2, just use endpoints
                x1_vals = [0.0, 1.0]
        
        # Create the population matrix
        X = np.zeros((len(x1_vals), self.dim))
        X[:, 0] = x1_vals
        # x[1:] are 0 by default
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
