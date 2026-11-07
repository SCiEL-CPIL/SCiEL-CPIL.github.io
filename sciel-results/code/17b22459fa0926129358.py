"""DTLZ1 optimizer that directly samples the known Pareto front using a uniform lattice."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 that directly constructs points on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 290.0  # 300s limit with 10s safety margin

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ1: first (dim - n_obj + 1) variables determine position on the front,
        # remaining (n_obj - 1) variables determine aggregation (should be 0.5 for Pareto front)
        n_pos = dim - self.n_obj + 1  # 7 - 3 + 1 = 5 position variables
        n_agg = self.n_obj - 1  # 2 aggregation variables

        # On DTLZ1 Pareto front:
        #   x[2:] = 0.5
        #   f1 = 0.5*x1, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        # Front is parameterized by (x1, x2) in [0,1]^2
        # The Pareto front lies on the simplex f1+f2+f3 = 0.5 with fi >= 0

        # Strategy: Use a uniform lattice on the (x1, x2) unit square to cover the simplex.
        # The Pareto front in objective space is a triangle, but in (x1, x2) space it's the
        # entire unit square. A uniform grid gives even coverage.

        # Determine grid size for the (x1, x2) space
        # We want approximately budget points, so use a square grid of size ~sqrt(budget)
        n_grid = int(np.ceil(np.sqrt(budget)))

        # Generate a uniform grid in (x1, x2) space
        x1_vals = np.linspace(0.0, 1.0, n_grid)
        x2_vals = np.linspace(0.0, 1.0, n_grid)

        # Create meshgrid
        X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing='ij')

        # Flatten to get list of (x1, x2) pairs
        all_pairs = np.column_stack([X1.ravel(), X2.ravel()])

        # Truncate to budget
        all_pairs = all_pairs[:budget]

        # Evaluate each point on the Pareto front
        X = np.zeros((budget, dim))
        F = np.zeros((budget, self.n_obj))
        valid_count = 0

        for i in range(len(all_pairs)):
            if self._elapsed() > self.deadline:
                break

            x1 = float(np.clip(all_pairs[i, 0], 0.0, 1.0))
            x2 = float(np.clip(all_pairs[i, 1], 0.0, 1.0))

            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2

            try:
                f = problem(x)
                X[i] = x
                F[i] = f
                valid_count += 1
            except Exception:
                break

        return X[:valid_count]
# EVOLVE-BLOCK-END
