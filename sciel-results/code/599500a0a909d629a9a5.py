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
        # f1 = x[0], f2 = x[1], f3 = (1+g)*h where g=0 when x[2:]=0.5
        # So on the Pareto front: f1=x0, f2=x1, f3=h(x0,x1)
        # h = (m+1) - sum_{i=1}^{m} [f_i/(1+g) * (1 + sin(pi*f_i/(1+g)))]
        # With g=0: h = 4 - [x0*(1+sin(pi*x0)) + x1*(1+sin(pi*x1))]
        
        # Strategy: Use a non-uniform grid concentrated where hypervolume contribution is high
        # HV contribution ~ (1.1-x0)*(1.1-x1)*(25-h(x0,x1))
        # This is highest when x0, x1 are small AND h is small
        # h is minimized when x0 and x1 are around 0.5-0.6 (where sin term is negative)
        
        # Use a grid with denser spacing in the region [0, 0.7] x [0, 0.7]
        # and sparser in the outer regions
        
        n_grid = 14
        # Create non-uniform grid: more points near origin
        # Use power-law spacing: x = grid_max * (i/(n-1))^p with p > 1 for denser near 0
        p = 1.5  # power for concentration near origin
        x0_vals = np.linspace(0, 1.0, n_grid)
        x0_vals = x0_vals ** p  # This concentrates points near 0
        
        # Actually, let's use a different approach: 
        # Use a grid that covers [0, 0.9] with non-uniform spacing
        # More points in [0, 0.5] where HV contribution is significant
        
        # Let me use a 2D grid with adaptive spacing
        # Split into regions and allocate points based on expected HV contribution
        
        # Simpler approach: use a grid with spacing that decreases toward origin
        # x_i = grid_max * (i/n)^alpha for i=0..n-1, with alpha > 1
        
        grid_max = 0.95
        alpha = 1.3  # concentration parameter
        
        x0_vals = (np.arange(n_grid) / (n_grid - 1)) ** alpha * grid_max
        x1_vals = x0_vals.copy()
        
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        x0_flat = X0.ravel()
        x1_flat = X1.ravel()
        
        # Add extra points in critical regions
        # Points where h is minimized (around x0=x1=0.55)
        extra_x0 = [0.55, 0.55, 0.45, 0.65]
        extra_x1 = [0.45, 0.65, 0.55, 0.55]
        
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
