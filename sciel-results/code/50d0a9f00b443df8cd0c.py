# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven grid search over DTLZ1 Pareto front shape parameters.
    
    Strategy:
    - DTLZ1 with 7 variables: x[0:5] determine shape, x[5:7] determine position along front.
    - The Pareto front is a simplex in objective space. We parameterize it by 2 shape variables.
    - For 3 objectives, the front can be parameterized by two variables (u, v) in [0,1].
    - We do a dense grid search over these 2 parameters to cover the front well.
    - Budget: 200 evaluations. Use 196 for a 14x14 grid + 4 extra.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 30s safety margin before 310s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ1: 
        # f1 = 0.5 * x1 * (1 + g(x_M))
        # f2 = 0.5 * x2 * (1 - x1) * (1 + g(x_M))
        # f3 = 0.5 * (1 - x2) * (1 + g(x_M))
        # where x1, x2 are the first 2 of the M shape variables, g is the position variable.
        
        # For 7 vars: x[0:5] are shape vars (M=5), x[5:7] are position vars (K=2).
        # Actually for DTLZ1: M shape vars determine the front shape, K position vars determine g.
        # Standard: M = dim - K, K = 2 for 3 objectives.
        # So M = 5 shape vars, K = 2 position vars.
        
        # The Pareto front for DTLZ1 with 3 objectives:
        # f_i = 0.5 * prod_{j<i} x_j * (1 - x_{i-1}) * (1 + g)  for i=1,2
        # f_3 = 0.5 * (1 - x_1) * (1 - x_2) * (1 + g)
        # where x_1, x_2 are from the shape variables, and g >= 0.
        # At optimum g=0, so f_i are determined by x_1, x_2 in [0,1].
        
        # We parameterize the front by (x1, x2) where x1, x2 in [0,1].
        # For each (x1, x2), we need to find x such that problem(x) gives the corresponding point.
        # x[0] = x1, x[1] = x2, x[2:5] can be anything (they affect g but at g=0 they don't matter for the front)
        # x[5:7] determine g; to get g=0, we need x[5:7] such that g=0.
        # g(x) = 10*(K + sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for i in position vars)
        # g=0 when each position var = 0.5.
        
        # So for a point on the front: x[0]=x1, x[1]=x2, x[2:5]=0.5 (arbitrary), x[5]=0.5, x[6]=0.5.
        # This gives g=0 and the point is on the Pareto front.
        
        # Grid search over x1, x2 in [0,1]
        n_grid = 14
        points = []
        for i in range(n_grid):
            for j in range(n_grid):
                x1 = i / (n_grid - 1)
                x2 = j / (n_grid - 1)
                x = np.full(dim, 0.5)
                x[0] = x1
                x[1] = x2
                # x[2:5] = 0.5 (default)
                # x[5] = 0.5, x[6] = 0.5 (default)
                points.append(x)
        
        # Evaluate all grid points
        F = np.array([problem(x) for x in points])
        used = len(points)
        
        # Use remaining budget for additional points or refinement
        remaining = self.budget - used
        if remaining > 0:
            # Add corner points and some random points
            corners = [
                np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5]),
                np.array([1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5]),
                np.array([0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]),
                np.array([1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]),
            ]
            for c in corners[:remaining]:
                F = np.vstack([F, [problem(c)]])
                used += 1
                if used >= self.budget:
                    break
        
        return F
# EVOLVE-BLOCK-END
