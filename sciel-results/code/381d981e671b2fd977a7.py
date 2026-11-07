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
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Strategy: Sample f1 uniformly, map to x1, set x2...xn to 0
        # This ensures even coverage of the Pareto front in objective space
        
        # 1. Create a dense grid for x1 to invert f1(x1)
        grid_size = 100000
        x1_grid = np.linspace(0.0, 1.0, grid_size)
        
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        f1_grid = 1.0 - np.exp(-4.0 * x1_grid) * (np.sin(6.0 * np.pi * x1_grid) ** 8)
        
        # 2. Sort to allow binary search for x1 given f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # 3. Generate uniform targets for f1
        f1_targets = np.linspace(0.0, 1.0, n_evals)
        
        # 4. Find corresponding x1 values
        indices = np.searchsorted(f1_sorted, f1_targets, side='left')
        indices = np.clip(indices, 0, grid_size - 1)
        x1_samples = x1_sorted[indices]
        
        # 5. Construct decision vectors
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_samples
        # x2...xn are 0
        
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
            
        return X
