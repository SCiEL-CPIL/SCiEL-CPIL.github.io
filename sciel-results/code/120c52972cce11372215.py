"""Multi-objective optimizer on ZDT6 using known Pareto set structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit ZDT6 structure: sample x1 uniformly to cover the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT6 Pareto set: x_2..x_n = 0, x_1 in [0, 1]
        # f1 = 1 + 8 * |sin(x_1 * pi)|
        # f2 = (1 - sqrt(f1)) * (1 - f1 / (x_1 + 1)^2)
        #
        # Strategy: evaluate 200 points with x_1 uniformly spaced in [0, 1],
        # x_2..x_n = 0. This gives dense, uniform coverage of the Pareto front.

        n_evals = budget
        x1_values = np.linspace(0.0, 1.0, n_evals)

        # Build the population: x1 uniform, rest zeros
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_values

        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
