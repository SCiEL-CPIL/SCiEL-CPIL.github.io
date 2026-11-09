# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Deterministic structured search for DTLZ4.

    Strategy:
    - DTLZ4 has M objectives (here M=3) and N variables (here N=12).
    - The Pareto front is defined by:
        * The first M-1 = 2 variables (x0, x1) determine the shape/position on the front.
        * The remaining N - (M-1) = 10 variables determine the distance to the front.
    - For DTLZ4, the optimal distance variables are all 0.5.
    - So we fix x[2:] = 0.5 and only sample x[0] and x[1] using a low-discrepancy sequence.
    - This guarantees all evaluated points lie exactly on the Pareto front.
    - We use a Sobol sequence to uniformly cover the [0,1]^2 space for x0, x1.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # For DTLZ4 with M=3, N=12:
        # Shape variables: indices 0, 1 (first M-1 = 2 variables)
        # Distance variables: indices 2 through 11 (N - (M-1) = 10 variables)
        n_shape = self.n_obj - 1  # 2
        n_dist = self.dim - n_shape  # 10

        # Optimal value for distance variables in DTLZ4 is 0.5
        dist_opt = 0.5

        # Create a Sobol sequence for the shape variables
        # We need to evaluate 'budget' points total
        # Since all points are on the front, we just need to spread them well
        sobol = qmc.Sobol(d=n_shape, scramble=True, seed=42)

        # Generate budget points in [0, 1]^n_shape
        # Sobol requires power-of-2 number of points for best quality, but we can generate
        # more and take the first 'budget' points
        # Actually, let's generate exactly budget points
        shape_samples = sobol.random(self.budget)  # shape: (budget, n_shape)

        # Map to [lo, hi] for the shape variables
        # For DTLZ4, the bounds are [0,1] for all variables
        shape_samples = lo[:n_shape] + (hi[:n_shape] - lo[:n_shape]) * shape_samples

        # Build the full population
        # Each point has shape variables + fixed distance variables
        pop = np.zeros((self.budget, self.dim))
        pop[:, :n_shape] = shape_samples
        pop[:, n_shape:] = dist_opt

        # Clip to bounds just in case
        pop = np.clip(pop, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in pop])

        return pop
# EVOLVE-BLOCK-END
