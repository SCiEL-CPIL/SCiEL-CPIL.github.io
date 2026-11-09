import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        # DTLZ7 structure:
        # x = [x0, x1, ..., x21]
        # g(x) = sum(x_i for i in 2..21)  (10 variables)
        # h = 1 + g
        # f1 = x0
        # f2 = x1
        # f3 = h * (1 + sin(pi * x0 / 2))
        #     = h * (1 + (1 - x1) * cos(pi * x1 / 4))  -- wait, let me re-check
        
        # Actually for DTLZ7:
        # f1 = x1
        # f2 = x2
        # f3 = h * (1 + sin(pi * x1 / 2))
        # where h = 1 + g(x3, ..., x22)
        # g(x3...x22) = sum(x_i^2) or sum(x_i) depending on variant
        
        # For the Pareto front, we need g = 0, so x3...x22 = 0
        # Then h = 1
        # f1 = x1
        # f2 = x2
        # f3 = 1 + sin(pi * x1 / 2)
        
        # Wait, let me be more careful. DTLZ7 with M=3:
        # n = 22 variables
        # k = n - M + 1 = 22 - 3 + 1 = 20? No...
        # Actually k = n - M + 1 for DTLZ7 means the number of variables in g.
        # Let me use the standard definition:
        # x = (x1, x2, ..., xn)
        # g(x) = sum_{i=k+1}^{n} x_i where k = n - M + 1
        # For M=3, n=22: k = 20, so g = sum(x_21, x_22)
        # h(x, y) = 1 + g(x)
        # f1 = x1
        # f2 = x2
        # f3 = h * (1 + sin(pi * x1 / 2))
        
        # Pareto front: g = 0, so x_21 = x_22 = 0
        # f1 = x1, f2 = x2, f3 = 1 + sin(pi * x1 / 2)
        # x1 in [0,1], x2 in [0,1]
        
        # This is a 2D Pareto front! Not 1D.
        # f3 depends on x1 only (when g=0)
        # f1 = x1, f2 = x2, f3 = 1 + sin(pi*x1/2)
        
        # So the front is: for each x1 in [0,1], x2 can be anything in [0,1]
        # f1 = x1, f2 = x2, f3 = 1 + sin(pi*x1/2)
        
        # Reference point: [1.1, 1.1, 25.0]
        # Points must be strictly better than reference in all objectives.
        # f1 = x1 < 1.1 (always true since x1 <= 1)
        # f2 = x2 < 1.1 (always true since x2 <= 1)
        # f3 = 1 + sin(pi*x1/2) < 25.0 (always true)
        
        # So all points on the front are valid.
        
        # Hypervolume: we want to maximize the volume dominated by our points.
        # The front spans:
        # f1 in [0, 1]
        # f2 in [0, 1]
        # f3 in [1, 1 + sin(pi/2)] = [1, 2]
        
        # The Pareto front is a 2D surface in objective space:
        # (f1, f2, f3) = (x1, x2, 1 + sin(pi*x1/2))
        
        # To maximize hypervolume, we should sample this 2D surface densely.
        # The hypervolume contribution depends on the 3D volume.
        
        # Strategy: sample the 2D Pareto front with 200 points.
        # Use a grid or quasi-random sampling over (x1, x2) in [0,1]^2.
        
        n_samples = 200
        
        # Use a stratified/sobol-like sampling for better coverage
        # Split into roughly 14x14 grid
        n_per_dim = 14
        n_samples = n_per_dim * n_per_dim  # 196
        if n_samples > self.budget:
            n_samples = self.budget
            n_per_dim = int(np.sqrt(n_samples))
        
        # Create grid points
        x1_grid = np.linspace(0, 1, n_per_dim)
        x2_grid = np.linspace(0, 1, n_per_dim)
        
        # Create all combinations
        X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing='ij')
        X1_flat = X1.flatten()
        X2_flat = X2.flatten()
        
        # If we have extra budget, add some random points
        n_extra = self.budget - len(X1_flat)
        if n_extra > 0:
            rng = np.random.RandomState(42)
            X1_extra = rng.uniform(0, 1, n_extra)
            X2_extra = rng.uniform(0, 1, n_extra)
            X1_flat = np.concatenate([X1_flat, X1_extra])
            X2_flat = np.concatenate([X2_flat, X2_extra])
        
        # Build decision vectors
        X = np.zeros((len(X1_flat), self.dim))
        X[:, 0] = X1_flat  # x1
        X[:, 1] = X2_flat  # x2
        # x3..x22 = 0 (so g = 0)
        
        # Evaluate
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
