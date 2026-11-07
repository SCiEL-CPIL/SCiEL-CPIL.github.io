import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin (300s limit, keep 20s buffer)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        dim = self.dim
        
        # Strategy:
        # 1. Coarse grid search on x_1 with x_2..x_n = 0
        # 2. Identify top K candidates (by low f_2, but also spread in x_1)
        # 3. Fine grid search around each candidate
        
        n_coarse = 100
        K = 10
        n_fine_per_candidate = (n_evals - n_coarse) // K
        
        if n_fine_per_candidate < 1:
             n_fine_per_candidate = 1
             K = (n_evals - n_coarse) // n_fine_per_candidate
             if K < 1:
                 K = 1
                 n_fine_per_candidate = n_evals - n_coarse

        # 1. Coarse Search
        X_coarse = np.zeros((n_coarse, dim))
        x1_coarse = np.linspace(lo[0], hi[0], n_coarse)
        X_coarse[:, 0] = x1_coarse
        X_coarse[:, 1:] = lo[1:]
        
        F_coarse = np.array([problem(x) for x in X_coarse])
        
        # Check time after coarse
        if time.time() > deadline:
            return None
        
        # 2. Find Top K
        # We want the lowest f_2 values
        # Note: f_1 is x_1, so we just need to keep track of x_1
        # Sort by f_2
        sorted_indices = np.argsort(F_coarse[:, 1])[:K]
        
        X_fine = []
        F_fine = []
        
        for idx in sorted_indices:
            x_center = X_coarse[idx, 0]
            f_center = F_coarse[idx]
            
            # Define a small interval around x_center
            # The width of the coarse grid step
            delta = (hi[0] - lo[0]) / n_coarse
            
            # Sample n_fine_per_candidate points in [x_center - delta, x_center + delta]
            # Clip to bounds
            x_low = max(lo[0], x_center - delta)
            x_high = min(hi[0], x_center + delta)
            
            if x_high <= x_low:
                continue
                
            x1_fine = np.linspace(x_low, x_high, n_fine_per_candidate)
            
            for x1 in x1_fine:
                x_trial = np.zeros(dim)
                x_trial[0] = x1
                x_trial[1:] = lo[1:]
                
                f_trial = problem(x_trial)
                X_fine.append(x_trial)
                F_fine.append(f_trial)
                
                # Check budget
                if len(F_coarse) + len(F_fine) >= n_evals:
                    break
                
                # Check time
                if time.time() > deadline:
                    return None
            
            if len(F_coarse) + len(F_fine) >= n_evals:
                break
                
        return None
