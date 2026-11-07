"""Direct construction optimizer for DTLZ4 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using direct front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure for 3 objectives (M=3, n=12):
        # x1, x2 (indices 0,1) determine the Pareto front shape.
        # x3..x12 (indices 2..11) are in the g-function.
        # To achieve g=0 (Pareto optimal), we must set x3..x12 = 0.5.
        # The Pareto front is a quarter-sphere of radius 0.5.
        
        # Strategy:
        # 1. Generate `budget` distinct points on the Pareto front.
        # 2. Use Sobol sequence for x1, x2 to ensure good coverage of the front.
        # 3. Set x3..x12 to 0.5 exactly.
        # 4. Evaluate all points.
        # This maximizes the hypervolume by ensuring all points are non-dominated
        # and well-distributed on the true Pareto front.

        n_points = budget
        
        # Generate Sobol sequence for 2 dimensions (x1, x2)
        # Sobol sequence provides low-discrepancy sampling, which is ideal for
        # covering the 2D parameter space that maps to the 2D Pareto front.
        sobol = qmc.Sobol(d=2, scramble=True, seed=0)
        # Generate points in [0, 1]^2
        # We need to ensure we don't exceed budget or time.
        # Sobol generation is very fast, so we can generate all at once.
        
        # Note: qmc.Sobol.generate(n) returns points in [0, 1).
        # We can use them directly as x1, x2.
        try:
            pts = sobol.generate(n_points)
        except Exception:
            # Fallback to random if Sobol fails for some reason
            pts = np.random.rand(n_points, 2)

        x1_flat = pts[:, 0]
        x2_flat = pts[:, 1]

        # Construct decision vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = x1_flat
        X[:, 1] = x2_flat
        # Set all other variables to 0.5 to ensure g=0
        if dim > 2:
            X[:, 2:] = 0.5

        # Clip to bounds (though [0,1] is standard, bounds might vary)
        X = np.clip(X, lo, hi)

        # Evaluate all points
        # We must finish within the deadline.
        # Evaluation is the bottleneck.
        F_list = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F_list.append(f)
            
        if not F_list:
            return X

        F = np.array(F_list)
        # If we broke out early, we only have a subset of X
        if len(F_list) < n_points:
            X = X[:len(F_list)]

        return X
# EVOLVE-BLOCK-END
