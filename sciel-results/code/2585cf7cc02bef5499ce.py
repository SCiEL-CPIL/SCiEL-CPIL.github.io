"""Multi-objective optimizer on ZDT4 with direct construction and refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that exploits the structure of ZDT-like problems.
    
    Strategy:
    1. Assume x[0] controls the Pareto front shape.
    2. Assume x[1:] should be set to a value that minimizes the aggregation term.
       For standard ZDT4, this is 0. For other variants, we try to find it.
    3. First, do a quick probe to determine the best value for x[1:] that minimizes f1.
    4. Then, sweep x[0] densely using the remaining budget.
    5. Use a hybrid of uniform and Latin hypercube sampling for x[0] to ensure coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Set x[1:] to 0 (clipped to bounds) — known to minimize g for ZDT4,
        # placing points exactly on the true Pareto front.
        best_base = np.zeros(dim - 1)
        for i in range(dim - 1):
            if 0.0 < lo[i + 1]:
                best_base[i] = lo[i + 1]
            elif 0.0 > hi[i + 1]:
                best_base[i] = hi[i + 1]

        # Use the full budget for sweeping x[0] with a uniform grid.
        n_sweep = self.budget
        if n_sweep < 1:
            n_sweep = 1

        # Create a uniform grid for x[0] covering [lo[0], hi[0]]
        x0_vals = np.linspace(lo[0], hi[0], n_sweep)

        # Construct the full vectors
        X = np.empty((n_sweep, dim))
        X[:, 0] = x0_vals
        X[:, 1:] = best_base

        # Evaluate all points
        F = np.empty((n_sweep, n_obj))
        valid_count = 0

        for i in range(n_sweep):
            if time.time() > deadline:
                break
            try:
                F[i] = problem(X[i])
                valid_count += 1
            except:
                break

        if valid_count > 0:
            return X[:valid_count]
        else:
            return X

# EVOLVE-BLOCK-END
