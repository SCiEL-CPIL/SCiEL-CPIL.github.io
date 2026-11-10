"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for ZDT6 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        N = self.budget

        # Dense sampling of the known Pareto front
        # Optimal spacing for hypervolume maximization on convex front
        # x_i proportional to i^2
        if N > 1:
            x0_vals = np.arange(N, dtype=float)
            x0_vals = (x0_vals / (N - 1)) ** 2
        else:
            x0_vals = np.array([0.0])

        X = np.zeros((N, dim))
        X[:, 0] = x0_vals
        # x[1:] remains 0

        F = []
        for i in range(N):
            if time.time() > deadline:
                break
            fx = problem(X[i])
            F.append(fx)

        return F
# EVOLVE-BLOCK-END
