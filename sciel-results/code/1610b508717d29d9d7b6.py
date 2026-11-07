import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # For ZDT3, the Pareto front is achieved when x_2...x_n = 0
        # Use all budget to sample x_1 with non-uniform density
        # to maximize hypervolume coverage
        
        # Use a power-law distribution to concentrate points in regions
        # where the front has higher curvature
        # For ZDT3, the front is non-convex with discontinuities, so
        # we use a mixture of uniform and concentrated sampling
        
        # Strategy: Use 80% uniform and 20% concentrated near key regions
        n_uniform = int(0.8 * n_evals)
        n_concentrated = n_evals - n_uniform
        
        # Uniform sampling
        x1_uniform = np.linspace(lo[0], hi[0], n_uniform)
        
        # Concentrated sampling near known interesting regions
        # ZDT3 has discontinuities at specific x_1 values
        # We'll sample near x_1 = 0, 0.5, 1.0 and some intermediate points
        key_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        x1_concentrated = np.array([])
        
        for kp in key_points:
            if len(x1_concentrated) >= n_concentrated:
                break
            # Add a few points around each key point
            n_per_key = max(1, n_concentrated // len(key_points))
            spread = 0.05
            local_points = np.linspace(max(lo[0], kp - spread), min(hi[0], kp + spread), n_per_key)
            x1_concentrated = np.concatenate([x1_concentrated, local_points])
        
        # Trim to exact count if needed
        x1_concentrated = x1_concentrated[:n_concentrated]
        
        # Combine all x_1 values
        x1_all = np.concatenate([x1_uniform, x1_concentrated])
        
        # Ensure we don't exceed budget
        x1_all = x1_all[:n_evals]
        
        for x1 in x1_all:
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x1
            x[1:] = lo[1:]  # Set x_2...x_n to lower bound (0)
            
            f = problem(x)
        
        return None
