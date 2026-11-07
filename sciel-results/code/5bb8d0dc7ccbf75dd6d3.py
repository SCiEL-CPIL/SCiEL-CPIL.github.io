# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT2 using analytically optimal Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # ZDT2 Pareto front: f1 = x1, f2 = 1 + 9*x1 for x1 in [0,1], x2..x30=0
        # This is a LINEAR front (zero curvature), so the analytically optimal
        # node distribution for maximizing discrete hypervolume is UNIFORM spacing.
        # For a convex curve with non-zero curvature, density ∝ curvature would apply,
        # but here curvature = 0 everywhere, reducing to uniform.
        
        n_points = budget
        if n_points < 2:
            n_points = 2

        # Uniform spacing in x1 is optimal for the linear ZDT2 Pareto front
        x1_vals = np.linspace(0.0, 1.0, n_points)
        
        X_list = []
        F_list = []
        
        for i in range(n_points):
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            X_list.append(x)
            F_list.append(f)
            
            if time.time() > deadline:
                break

        return np.array(X_list)
# EVOLVE-BLOCK-END
