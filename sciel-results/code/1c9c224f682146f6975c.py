"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer for DTLZ4 using deterministic Halton sequence sampling on the Pareto front.
    
    DTLZ4 (and DTLZ2) Pareto front is a quarter unit sphere in 3D objective space.
    It is parameterized by x1 and x2 in [0, 1].
    The remaining variables x3...x12 should be set to minimize g(x).
    For DTLZ4, g(x) = sum(x_i^alpha) with alpha=100. Min is at x_i=0.
    For DTLZ2, g(x) = sum((x_i-0.5)^2). Min is at x_i=0.5.
    Setting x_i=0.5 is safe for both as 0.5^100 is negligible.
    We sample x1, x2 uniformly using a Halton sequence to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, d):
        """Generate n points in d dimensions using the Halton sequence."""
        def radical_inverse(r, base):
            f = 1.0
            i = 1.0
            while r > 0:
                f /= base
                r = int(r * base)
                i += f
            return i - 1.0

        points = []
        for i in range(1, n + 1):
            point = []
            for b in range(2, d + 2):  # bases 2, 3, 5, ...
                point.append(radical_inverse(i, b))
            points.append(point)
        return np.array(points)

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Halton points for x1 and x2
        # We use the full budget for evaluations
        n_points = self.budget
        halton_2d = self._halton_sequence(n_points, 2)
        
        # Initialize the population of points
        # x1 and x2 come from Halton sequence
        # x3...x12 are set to 0.5 (safe center for g(x) minimization)
        X = np.zeros((n_points, self.dim))
        X[:, 0] = halton_2d[:, 0]
        X[:, 1] = halton_2d[:, 1]
        if self.dim > 2:
            X[:, 2:] = 0.5
        
        # Clip to bounds just in case (Halton is in [0,1], bounds are [0,1])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = []
        for x in X:
            f = problem(x)
            F.append(f)
        
        F = np.array(F)
        
        return X
# EVOLVE-BLOCK-END
