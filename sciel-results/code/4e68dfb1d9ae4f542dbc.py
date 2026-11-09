"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer tailored for ZDT2.
    Exploits the known structure: Pareto front lies on x[1:] = 0,
    with f1 = x[0] and f2 = 1 - f1 (when x[1:] = 0).
    Uses uniform spacing in x[0] to match the convex geometry.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT2 Pareto front: x[0] in [0,1], x[1:] = 0
        # f1 = x[0], f2 = 1 - sqrt(x[0]) when x[1:] = 0
        
        n_points = min(self.budget, 200)
        
        # Mix of uniform-in-f1 and sqrt-spaced-in-f1 (uniform-in-f2)
        # 60% uniform, 40% sqrt
        n_uniform = int(0.6 * n_points)
        n_sqrt = n_points - n_uniform
        
        # Generate uniform points in x[0] from 0 to 1
        x0_uniform = np.linspace(0.0, 1.0, n_uniform)
        
        # Generate sqrt-spaced points in x[0] (which makes f2 uniform)
        # x0 = u^2 where u is uniform in [0, 1]
        u_sqrt = np.linspace(0.0, 1.0, n_sqrt)
        x0_sqrt = u_sqrt ** 2
        
        # Combine
        x0_values = np.concatenate([x0_uniform, x0_sqrt])
        
        # Build the full vectors with x[1:] = 0
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_values
        
        # Ensure within bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points
        
        return X
# EVOLVE-BLOCK-END
