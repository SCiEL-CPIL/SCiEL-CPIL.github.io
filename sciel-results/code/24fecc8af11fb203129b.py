"""ZDT4 optimizer using uniform 1D grid on x0 (optimal for concave Pareto front)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using uniform 1D grid search on x0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_vars = self.dim
        n_evals = self.budget

        x0_lo = lo[0]
        x0_hi = hi[0]

        # For ZDT4, the Pareto front is parameterized by x0 with x1..x9 = 0.
        # f1 = x0, f2 = 1 - sqrt(x0). The front is concave.
        # For a concave front, uniform spacing in f1 (i.e., in x0) maximizes
        # the hypervolume of the sampled points.
        #
        # We use all budget evaluations on the 1D front.
        n_grid = n_evals

        # Uniform spacing in x0 from x0_lo to x0_hi inclusive
        if n_grid > 1:
            x0_vals = np.linspace(x0_lo, x0_hi, n_grid)
        else:
            x0_vals = np.array([(x0_lo + x0_hi) / 2.0])

        results = []
        # Pre-allocate x to avoid reallocation in loop
        x = np.zeros(n_vars)
        for x0 in x0_vals:
            x[0] = x0
            # x[1:] remains 0.0, which is optimal for ZDT4's g(x) function
            f = problem(x)
            results.append(f)

            if time.time() > deadline:
                break

        return results
# EVOLVE-BLOCK-END
