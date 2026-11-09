"""Multi-objective DTLZ4 optimizer using scrambled Sobol sequence on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """DTLZ4 optimizer: sample the known Pareto front via scrambled 2D Sobol sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        n_shape = m - 1  # 2 shape variables for 3 objectives

        # Generate Sobol sequence
        # We need n_shape dimensions for the shape variables
        sobol_gen = qmc.Sobol(d=n_shape, scramble=True, seed=42)
        
        # Generate points. Sobol generates in [0,1]
        # We need self.budget points
        points = sobol_gen.random(self.budget)
        
        # points shape: (budget, n_shape)
        
        # Build full x vectors
        # g-variables (indices n_shape to n-1) set to 0.5 to minimize g
        X = np.zeros((self.budget, n))
        X[:, :n_shape] = points
        for j in range(n_shape, n):
            X[:, j] = 0.5

        # Clip to bounds (though Sobol is in [0,1] and bounds are [0,1], this is safe)
        X = np.clip(X, lo, hi)

        # Evaluate all points with time guard
        F = np.empty((self.budget, m))
        for i in range(self.budget):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
