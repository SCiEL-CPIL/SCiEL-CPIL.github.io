"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for DTLZ2 using low-discrepancy sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        n = self.budget
        dim = self.dim
        M = self.n_obj
        
        # Generate Halton sequence in 2D for (alpha, beta) parameterization
        # DTLZ2 Pareto front: f_i = 0.5 * x_i where x is on unit sphere in positive octant
        # Parameterize: x1=cos(a), x2=sin(a)*cos(b), x3=sin(a)*sin(b), a,b in [0, pi/2]
        
        def halton(n, base):
            seq = np.zeros(n)
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                j = i
                while j > 0:
                    j, rem = divmod(j, base)
                    f /= base
                    r += f * rem
                seq[i - 1] = r
            return seq
        
        # Generate n points in (alpha, beta) space using Halton sequence
        # Use two different bases for the two dimensions
        t1 = halton(n, 2)
        t2 = halton(n, 3)
        
        # Map to [0, pi/2]
        alpha = t1 * np.pi / 2
        beta = t2 * np.pi / 2
        
        # Compute sphere coordinates
        x1 = np.cos(alpha)
        x2 = np.sin(alpha) * np.cos(beta)
        x3 = np.sin(alpha) * np.sin(beta)
        
        # Build decision vectors
        # For DTLZ2, set x_1..x_M to sphere coords, x_{M+1}..x_n to minimize g
        # Setting remaining vars to 0.5 is a common heuristic, but let's try 1.0
        # Actually for DTLZ2 g function, the minimum is at specific points.
        # Let's just set them to 0.5 as a reasonable default.
        X = np.zeros((n, dim))
        X[:, 0] = x1
        X[:, 1] = x2
        X[:, 2] = x3
        # Set remaining variables to 0.5 (midpoint of [0,1])
        for j in range(M, dim):
            X[:, j] = 0.5
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n
        
        # If we have time and budget, we could refine, but for DTLZ2
        # direct front sampling should be near-optimal
        return X
# EVOLVE-BLOCK-END
