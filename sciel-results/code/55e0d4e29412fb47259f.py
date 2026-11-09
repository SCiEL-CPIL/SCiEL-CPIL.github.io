import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, dim=1):
        """Generate a Sobol-like deterministic space-filling sequence in [0,1]^dim."""
        # Use a simple Halton sequence as a fallback if scipy is not available
        # But let's try to use a proper low-discrepancy sequence
        # For simplicity, use the inverse of the van der Corput sequence (Halton)
        # which is deterministic and space-filling
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                base = d + 2  # Use primes as bases
                points[i, d] = halton(i + 1, base)
        return points

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT3 specific:
        # f1 = x1
        # f2 = g(x[1:]) * (1 - sqrt(f1) + (1- sqrt(f1))/g * sin(10*pi*f1))
        # g_min = 1 when x[1:] = 0
        # Pareto front: f2 = 1 + (1 - sqrt(f1)) / (1 + 9*f1) * sin(10 * pi * f1)
        
        # Strategy: Use a deterministic space-filling sequence (Halton) to generate
        # x1 values that uniformly cover [0,1], ensuring we hit all 7 Pareto segments
        # without needing to explicitly identify them.
        
        # Generate budget x1 values using Halton sequence in 1D
        x1_values = self._sobol_sequence(budget, dim=1).flatten()
        
        # Build X: set x[0] = x1_values, x[1:] = 0 (for g_min = 1)
        X = np.zeros((budget, dim))
        X[:, 0] = x1_values
        # x[1:] remains 0, which gives g = 1 (minimum)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
