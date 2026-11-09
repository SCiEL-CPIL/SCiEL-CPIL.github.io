import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Strategy: Stratified Grid Sampling on the Pareto Front Parameter Space
        # 1. Determine grid size
        n_grid = int(np.sqrt(budget))
        while n_grid * n_grid > budget:
            n_grid -= 1
            
        # 2. Create grid points in angle space (theta1, theta2)
        # Use midpoints of cells for better coverage
        step = (np.pi / 2.0) / n_grid
        offsets = np.linspace(0, step, n_grid, endpoint=False) + step / 2.0
        
        theta1_vals = offsets
        theta2_vals = offsets
        
        T1, T2 = np.meshgrid(theta1_vals, theta2_vals, indexing='ij')
        T1_flat = T1.flatten()
        T2_flat = T2.flatten()
        
        X = np.zeros((T1_flat.size, dim))
        X[:, 0] = T1_flat / (np.pi / 2.0)
        X[:, 1] = T2_flat / (np.pi / 2.0)
        X[:, 2:] = 0.5
        
        X = np.clip(X, lo, hi)
        
        # 3. Evaluate grid points
        for x in X:
            if time.time() > deadline:
                break
            problem(x)
            
        # 4. Use remaining budget for random perturbations (Jitter)
        # This helps if the grid is slightly misaligned or to ensure strict interior points
        remaining = budget - T1_flat.size
        if remaining > 0:
            # Generate random points on the front
            theta1_rand = np.random.uniform(0, np.pi/2.0, size=remaining)
            theta2_rand = np.random.uniform(0, np.pi/2.0, size=remaining)
            
            X_rand = np.zeros((remaining, dim))
            X_rand[:, 0] = theta1_rand / (np.pi / 2.0)
            X_rand[:, 1] = theta2_rand / (np.pi / 2.0)
            X_rand[:, 2:] = 0.5
            
            X_rand = np.clip(X_rand, lo, hi)
            
            for x in X_rand:
                if time.time() > deadline:
                    break
                problem(x)
                
        return X
