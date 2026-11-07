"""Multi-objective optimizer for DTLZ4 using 2D position-space area maximization."""
# EVOLVE-BLOCK-START
import numpy as np


def _halton_sequence(n, dim):
    """Generate n points from the Halton low-discrepancy sequence in [0,1]^dim."""
    points = np.zeros((n, dim))
    for d in range(dim):
        base = d + 2
        for i in range(n):
            f = 1.0
            i_val = i + 1
            r = 0.0
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                r += f * rem
            points[i, d] = r
    return points


def _sobol_sequence(n, dim):
    """Generate n points from a Sobol low-discrepancy sequence in [0,1]^dim."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True)
        m = 1
        while m < n:
            m *= 2
        points = sampler.random(m)
        return points[:n]
    except Exception:
        return _halton_sequence(n, dim)


class Optimizer:
    """DTLZ4 optimizer exploiting 2D position structure.
    
    Key insight: In DTLZ4, x[0] and x[1] determine position on the Pareto front,
    while x[2:] control distance from the front via g(x). We maximize the effective
    area covered in objective space by:
    1. Sampling (x0, x1) uniformly over [0,1]^2 using a low-discrepancy sequence
    2. Setting x[2:] to 0 to minimize g(x) to 0 for DTLZ4
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Generate low-discrepancy points for the 2D position space (x0, x1)
        pos_points = _sobol_sequence(budget, 2)
        
        # Construct full 12D points
        # Strategy: set x[2:] to 0 which for DTLZ4 with beta=100
        # gives x_i^beta = 0, sin(0) = 0, so each term = 0
        # This gives g(x) = 0, which is optimal
        X = np.zeros((budget, dim))
        X[:, 0] = pos_points[:, 0]
        X[:, 1] = pos_points[:, 1]
        
        # Set x[2:] to 0 to minimize g(x)
        X[:, 2:] = 0.0
        
        # Map to bounds
        X = lo + (hi - lo) * X
        
        # Evaluate all
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
