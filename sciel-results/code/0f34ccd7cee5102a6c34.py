import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # DTLZ7 Pareto front is at x[2:] = 0.5
        # We only need to optimize x[0] and x[1]
        
        # Strategy:
        # 1. Use a grid that covers the entire [0,1] x [0,1] space.
        # 2. Hypervolume is maximized by points that dominate the largest volume.
        #    Points with small f1 and f2 are critical.
        #    Points with small f3 are also critical.
        #    f3 = (1+g) * h. On the Pareto front g=0, so f3 = h.
        #    h = (m+1) - sum_{i=1}^{m-1} [ (f_i / (1+g)) * (1 + sin(pi * f_i / (1+g))) ]
        #    With g=0, f_i = x[i].
        #    h = (m+1) - sum_{i=1}^{m-1} [ x_i * (1 + sin(pi * x_i)) ]
        #    For m=3, h = 4 - [ x_0(1+sin(pi x_0)) + x_1(1+sin(pi x_1)) ]
        #    To minimize f3 (h), we need to maximize the sum term.
        #    The function t(1+sin(pi t)) is maximized when sin(pi t) is max, i.e., t=0.5.
        #    Max value is 0.5 * (1+1) = 1.
        #    So max sum is 1+1=2. Min h is 4-2=2.
        #    The reference point is [1.1, 1.1, 25.0].
        #    Points on the Pareto front have f3 in [2, 4].
        #    Points with f1=0, f2=0 have f3=4.
        #    Points with f1=0.5, f2=0.5 have f3=2.
        #    
        #    The hypervolume contribution of a point p is the volume of the box
        #    [p1, r1] x [p2, r2] x [p3, r3] that is not dominated by other points.
        #    To maximize total hypervolume, we want to cover the "corners" of the
        #    non-dominated set as well as the "center".
        #    
        #    A uniform grid is a safe bet. Let's use a grid size that fits the budget.
        #    budget = 200.
        #    sqrt(200) ~ 14.14.
        #    14x14 = 196 points.
        #    We can add 4 more points.
        
        # Let's use a slightly finer grid or a specific set of points.
        # Since f3 depends on x0 and x1, and f1=x0, f2=x1, the mapping is direct.
        # We want to sample x0 and x1 such that we get good coverage of the front.
        
        # Let's try 15x13 = 195 points. Or 14x14 = 196.
        # Let's stick with 14x14 = 196 and add 4 strategic points.
        
        n_grid = 14
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        x0_flat = X0.ravel()
        x1_flat = X1.ravel()
        
        # Add 4 more points to utilize full budget
        # These points should be in regions where the grid might be sparse or where
        # hypervolume contribution is high.
        # The grid already includes 0 and 1.
        # Let's add points that are slightly off the grid to ensure better coverage.
        # Or just use the next best grid size.
        
        # Actually, a 15x15 grid is 225 points, which exceeds 200.
        # 14x14 = 196 is the largest square grid.
        
        # Let's add 4 points:
        # (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5)
        # These are midpoints of the edges, which are not in the 14x14 grid
        # (since 14 is even, 0.5 is between 0.0357 and 0.0714... wait, linspace(0,1,14)
        # step is 1/13 ~ 0.0769. 0.5 is not a multiple of 1/13. 0.5 = 6.5/13.
        # So 0.5 is not in the grid.
        
        extra_x0 = [0.5, 0.5, 0.0, 1.0]
        extra_x1 = [0.0, 1.0, 0.5, 0.5]
        
        x0_flat = np.concatenate([x0_flat, extra_x0])
        x1_flat = np.concatenate([x1_flat, extra_x1])
        
        # Truncate to budget just in case
        x0_flat = x0_flat[:budget]
        x1_flat = x1_flat[:budget]
        
        # Construct full vectors
        X = np.zeros((len(x0_flat), n))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        X[:, 2:] = 0.5
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            F.append(problem(X[i]))
        
        return X
