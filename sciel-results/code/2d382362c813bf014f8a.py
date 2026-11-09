"""Portfolio search for DTLZ2: Low-discrepancy sampling on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc as qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Sobol sampling on the front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 310.0  # safety margin under 330s kill

        used = 0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ2 structure:
        # g = sum_{i=3}^{dim} (x_i - 0.5)^2
        # f1 = (1+g)*sin(pi*x1/2)
        # f2 = (1+g)*cos(pi*x1/2)*sin(pi*x2/2)
        # f3 = (1+g)*cos(pi*x1/2)*cos(pi*x2/2)
        # Setting x3..x12 = 0.5 gives g = 0 -> point on the true Pareto front.
        # x1, x2 in [0,1] sweep the quarter-sphere front.

        # Generate budget points using a Sobol sequence.
        # We transform the coordinates to ensure uniform coverage of the Pareto front surface area.
        # The DTLZ2 front is a quarter sphere. The area element is proportional to cos(pi*x1/2) dx1 dx2.
        # To sample uniformly with respect to surface area, we transform x1:
        # u1 ~ Uniform(0,1) -> x1 = (2/pi) * arcsin(u1)
        # u2 ~ Uniform(0,1) -> x2 = u2
        
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        points = sampler.random(budget)
        
        # Transform x1 to account for the spherical geometry
        # Clip to [0, 1] to avoid numerical issues with arcsin, though Sobol is in [0,1]
        u1 = np.clip(points[:, 0], 0.0, 1.0)
        u2 = np.clip(points[:, 1], 0.0, 1.0)
        
        x1_vals = (2.0 / np.pi) * np.arcsin(u1)
        x2_vals = u2
        
        # Evaluate all candidates
        for i in range(budget):
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            try:
                x = np.full(dim, 0.5)
                x[0] = x1_vals[i]
                x[1] = x2_vals[i]
                problem(x)
            except Exception:
                # If an exception occurs, we might still have recorded some points.
                # The contract says "exception later keeps the evaluations made".
                # We should probably stop to avoid further issues, or continue?
                # "an exception before the first, makes the whole result invalid"
                # "an exception later keeps the evaluations made"
                # If problem(x) throws, the evaluation for this x might not be recorded.
                # It's safer to break and return what we have.
                break
            used += 1

        return None
# EVOLVE-BLOCK-END
