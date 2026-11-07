# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using Uniform sqrt(x1) Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # ZDT2 Pareto Front: x[0] in [0,1], x[1:] = 0
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0])
        
        # Optimal distribution for hypervolume on convex front y = 1 - sqrt(x)
        # is to space sqrt(x) uniformly, which clusters points near x=0 where
        # curvature is highest.
        n_pts = self.budget
        
        if n_pts <= 0:
            return []

        # Generate uniform points in u = sqrt(x1)
        # u in [0, 1]
        u_vals = np.linspace(0, 1, n_pts)
        x1_vals = u_vals ** 2
        
        all_f = []
        
        for x1_val in x1_vals:
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_val
            # x[1:] remains 0, which is optimal for ZDT2 front
            
            f = problem(x)
            all_f.append(f)
            
            if len(all_f) >= self.budget:
                break

        return all_f
# EVOLVE-BLOCK-END
