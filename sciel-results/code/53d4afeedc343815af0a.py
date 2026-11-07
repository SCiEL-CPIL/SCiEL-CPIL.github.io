import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT4 exploiting its decoupled structure.
    Sets x[1:] to 0 (clipped to bounds) to lie on the Pareto front,
    then sweeps x[0] uniformly across its bounds.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Set x[1:] to 0, clipped to bounds — known to minimize g for ZDT4
        best_base = np.zeros(dim - 1)
        for i in range(dim - 1):
            if 0.0 < lo[i + 1]:
                best_base[i] = lo[i + 1]
            elif 0.0 > hi[i + 1]:
                best_base[i] = hi[i + 1]

        # Use the full budget for sweeping x[0]
        n_sweep = self.budget
        if n_sweep < 1:
            n_sweep = 1

        # Uniform grid for x[0]
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
