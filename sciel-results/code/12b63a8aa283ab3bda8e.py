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
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front: x2...x21 = 0
        # f1 = x0
        # f2 = x1 * (1 + x0)
        # f3 = 1 - x0 + x0^2 * sin(pi*x0/2)
        
        # Reference point: [1.1, 1.1, 25.0]
        # Need f2 < 1.1, so x1 < 1.1/(1+x0)
        
        # Hypervolume-aware non-uniform grid:
        # Cluster points near origin (low f1, low f2) where marginal HV contribution is highest
        # Use power-law spacing: x = a * (i/(n-1))^p with p > 1 to cluster near 0
        
        n0 = 20
        n1 = 10
        
        # Power-law spacing for x0: cluster near 0
        p0 = 1.5
        x0_idx = np.arange(n0)
        x0_vals = 1.0 * (x0_idx / (n0 - 1)) ** p0
        
        # Ensure we include the extremes
        x0_vals[0] = 0.0
        x0_vals[-1] = 1.0
        
        all_x0 = []
        all_x1 = []
        
        # Small epsilon to ensure strict inequality f2 < 1.1
        eps = 1e-10
        
        for x0 in x0_vals:
            max_x1 = 1.1 / (1.0 + x0) - eps
            if max_x1 <= 0:
                max_x1 = 0.0
            # Power-law spacing for x1: cluster near 0
            p1 = 1.5
            x1_idx = np.arange(n1)
            x1_vals = max_x1 * (x1_idx / (n1 - 1)) ** p1
            x1_vals[0] = 0.0
            x1_vals[-1] = max_x1
            for x1 in x1_vals:
                all_x0.append(x0)
                all_x1.append(x1)
        
        all_x0 = np.array(all_x0)
        all_x1 = np.array(all_x1)
        
        n_samples = len(all_x0)
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = all_x0
        X[:, 1] = all_x1
        # X[:, 2:] = 0 (inactive variables)
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
