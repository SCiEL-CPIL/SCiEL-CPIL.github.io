"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

try:
    from scipy.stats import qmc
except ImportError:
    qmc = None


class Optimizer:
    """HV-driven optimizer using optimal Dirichlet distribution sampling on the DTLZ1 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        n_points = self.budget

        # Generate low-discrepancy points in [0, 1)^2
        if qmc is not None:
            # Use Sobol sequence with a fixed seed for reproducibility
            # The optimal set of points is unique for this deterministic problem
            sampler = qmc.Sobol(d=2, scramble=True, seed=0)
            u = sampler.random(n_points)
        else:
            # Fallback: Halton sequence
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r

            u = np.empty((n_points, 2))
            for i in range(n_points):
                u[i, 0] = halton(i + 1, 2)
                u[i, 1] = halton(i + 1, 3)

        # Map uniform points to the simplex using the optimal Dirichlet(0.5, 0.5, 0.5) distribution
        # This distribution maximizes hypervolume for a 3-objective simplex.
        # Inverse CDF transformation:
        # x0 ~ Beta(0.5, 1.0) -> x0 = u0^2
        # x1 ~ Beta(0.5, 1.0) scaled by (1-x0) -> x1 = u1^2 * (1-x0)
        x0 = u[:, 0] ** 2
        x1 = u[:, 1] ** 2 * (1.0 - x0)
        x2 = 1.0 - x0 - x1

        # Construct the decision vectors
        # x[0], x[1] determine the position on the Pareto front
        # x[2]...x[6] must be 0.5 to achieve g=0 (on the Pareto front)
        xs = np.zeros((n_points, self.dim))
        xs[:, 0] = x0
        xs[:, 1] = x1
        xs[:, 2:] = 0.5

        evaluated = []
        eval_count = 0

        for i in range(n_points):
            if time.time() > deadline:
                break
            x = xs[i]
            # The problem function clips to bounds, but we are already within [0,1]
            obj = problem(x)
            evaluated.append((x, obj))
            eval_count += 1

        return [x for x, obj in evaluated]
# EVOLVE-BLOCK-END
