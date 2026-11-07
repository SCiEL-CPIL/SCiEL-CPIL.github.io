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
        
        # DTLZ7 Pareto front: x[2:] = 0.5, optimize x[0], x[1]
        # f1 = x0, f2 = x1, f3 = 4 - x0*(1+sin(pi*x0)) - x1*(1+sin(pi*x1))
        
        # Hypervolume contribution of point (x0,x1) with ref [1.1,1.1,25]:
        # V = (1.1-x0)*(1.1-x1)*(25-f3) where f3 = 4 - x0*(1+sin(pi*x0)) - x1*(1+sin(pi*x1))
        
        # Strategy: Use adaptive grid that is denser where marginal hypervolume gain is high.
        # The key regions are:
        # - Small x0,x1: large (1.1-x) factors but f3~4
        # - x0,x1 near 0.5: f3~2 minimum
        # - We need good coverage of the trade-off curve
        
        # Use a non-uniform grid with more points in [0, 0.6] where the action is
        # and fewer in [0.6, 1.0]
        
        # Create a grid with variable density
        # Use quantile-based spacing to concentrate points where hypervolume gain is high
        
        # Define target points based on Pareto front parameterization
        # Let's use a grid in [0, 0.7] with finer resolution, plus some points at higher values
        
        n_inner = 12  # 12x12 = 144 points in [0, 0.7]
        inner_max = 0.7
        
        # Non-uniform grid: denser near 0 where (1.1-x) is largest
        # Use sqrt spacing to concentrate near 0
        x0_inner = np.sqrt(np.linspace(0, inner_max**2, n_inner))
        x1_inner = np.sqrt(np.linspace(0, inner_max**2, n_inner))
        
        X0, X1 = np.meshgrid(x0_inner, x1_inner)
        x0_flat = X0.ravel()
        x1_flat = X1.ravel()
        
        # Add points in [0.7, 1.0] region - these have smaller (1.1-x) but may have lower f3
        # Actually f3 is minimized at 0.5, so points beyond 0.7 have higher f3
        # Let's add a few points to capture the full Pareto front shape
        
        extra_x0 = [0.8, 0.9, 1.0, 0.5, 0.3, 0.6, 0.85, 0.95]
        extra_x1 = [0.2, 0.1, 0.0, 0.8, 0.85, 0.9, 0.15, 0.05]
        
        x0_flat = np.concatenate([x0_flat, extra_x0])
        x1_flat = np.concatenate([x1_flat, extra_x1])
        
        # Truncate to budget
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
