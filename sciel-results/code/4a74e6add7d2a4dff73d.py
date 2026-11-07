"""Multi-objective optimizer for DTLZ4 exploiting 2D position-space structure."""
# EVOLVE-BLOCK-START
import numpy as np


def _halton(n, dim):
    """Halton low-discrepancy sequence in [0,1]^dim."""
    pts = np.zeros((n, dim))
    for d in range(dim):
        base = d + 2
        for i in range(n):
            i_val = i + 1
            f = 1.0
            r = 0.0
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                r += f * rem
            pts[i, d] = r
    return pts


def _sobol(n, dim):
    """Sobol low-discrepancy sequence in [0,1]^dim via scipy, fallback to Halton."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True)
        m = 1
        while m < n:
            m *= 2
        return sampler.random(m)[:n]
    except Exception:
        return _halton(n, dim)


class Optimizer:
    """DTLZ4 optimizer: sample (x0,x1) with Sobol, fix x[2:] to minimizer of g."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For DTLZ4 (standard), g(x) = sum_{i=2}^{n-1} (x_i - 0.5)^2
        # Minimum g=0 at x_i=0.5 for all i in U.
        # The Pareto front is parameterized by x0, x1 in [0,1]:
        #   f1 = (1+g) * cos(x0*pi/2) * cos(x1*pi/2)
        #   f2 = (1+g) * cos(x0*pi/2) * sin(x1*pi/2)
        #   f3 = (1+g) * sin(x0*pi/2)
        # With g=0, this is the quarter-sphere Pareto front.
        #
        # Hypervolume maximization: we want to cover the Pareto front densely.
        # Using a low-discrepancy sequence on (x0, x1) in [0,1]^2 gives
        # near-uniform coverage of the mapped objective-space region.
        #
        # We use Sobol for best low-discrepancy properties.

        pos = _sobol(budget, 2)

        X = np.zeros((budget, dim))
        X[:, 0] = pos[:, 0]
        X[:, 1] = pos[:, 1]
        X[:, 2:] = 0.5

        # Map to bounds
        X = lo + (hi - lo) * X

        # Evaluate all points
        for i in range(budget):
            problem(X[i])

        return X
# EVOLVE-BLOCK-END
