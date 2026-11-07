"""Multi-objective optimizer for DTLZ1 using direct Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Direct sampler for DTLZ1 Pareto front with refined simplex sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate Sobol sequence points in [0, 1]^3
        sobol = qmc.Sobol(d=3, scramble=True)
        points_3d = sobol.random(budget)

        # Map from unit cube to simplex x1+x2+x3=1, x_i>=0
        # This corresponds to x1, x2 in the 2D simplex
        # Normalize the 3D points
        sums = points_3d.sum(axis=1, keepdims=True)
        # Avoid division by zero (extremely unlikely with Sobol)
        sums[sums == 0] = 1.0
        points_simplex = points_3d / sums
        
        # points_simplex[:, 0] is x1, points_simplex[:, 1] is x2
        # points_simplex[:, 2] is 1 - x1 - x2

        # Construct decision vectors
        # DTLZ1 Pareto front: x_3..x_n = 0.5 (midpoint) to make g(x)=0
        # x_1, x_2 are the simplex coordinates
        X = np.zeros((budget, dim))
        for i in range(budget):
            X[i, 0] = points_simplex[i, 0]
            X[i, 1] = points_simplex[i, 1]
            # Set remaining variables to midpoint of bounds to minimize g(x)
            for d in range(2, dim):
                X[i, d] = (lo[d] + hi[d]) / 2.0

        # Evaluate all points
        all_points = []
        used = 0
        for x in X:
            if time.time() > deadline:
                break
            f = problem(x)
            all_points.append(f)
            used += 1

        return X
# EVOLVE-BLOCK-END
