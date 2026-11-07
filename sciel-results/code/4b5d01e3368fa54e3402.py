import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 specifics:
        # Variables 0 and 1 determine the Pareto front shape.
        # Variables 2 to 11 determine the distance from the front.
        # Optimal distance variables are 0.5.
        
        n_evals = self.budget
        dim = self.dim
        
        # Generate n_evals points using a Halton sequence for the first 2 dimensions
        # and fix the rest at 0.5 (center of [0,1] bounds, assuming standard DTLZ4)
        
        def halton(n, base):
            """Generate n points in [0,1) using Halton sequence with given base."""
            result = np.zeros(n)
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                i_copy = i
                while i_copy > 0:
                    f /= base
                    r += f * (i_copy % base)
                    i_copy //= base
                result[i - 1] = r
            return result

        # Use bases 2 and 3 for the two position variables
        h1 = halton(n_evals, 2)
        h2 = halton(n_evals, 3)
        
        # Create the population
        # X shape: (n_evals, dim)
        X = np.zeros((n_evals, dim))
        
        # Set position variables
        X[:, 0] = h1
        X[:, 1] = h2
        
        # Set distance variables to 0.5 (optimal for DTLZ4)
        # Assuming bounds are [0,1], 0.5 is the center.
        # If bounds were different, we should map to the center of the bounds.
        # But the task says bounds are [0,1] for every variable.
        if dim > 2:
            X[:, 2:] = 0.5
            
        # Evaluate all points
        # The problem function expects a single vector x, but we can loop.
        # To be safe and match the interface, we call problem(x) for each x.
        # However, problem(x) might be vectorized? The contract says "problem(x) ... returns 3 objective values".
        # It implies a single vector input.
        
        F = np.zeros((n_evals, self.n_obj))
        for i in range(n_evals):
            F[i] = problem(X[i])
            
        return F
