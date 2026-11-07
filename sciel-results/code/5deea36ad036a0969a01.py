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
        
        # Strategy:
        # 1. Coarse grid search on x_1 with x_2..x_n = 0
        # 2. Identify top candidate in each of K bins to ensure coverage
        # 3. Fine grid search around each candidate
        
        n_coarse = 150
        K = 10
        n_fine_per_candidate = (n_evals - n_coarse) // K
        
        if n_fine_per_candidate < 1:
             n_fine_per_candidate = 1
             K = max(1, (n_evals - n_coarse) // n_fine_per_candidate)

        # 1. Coarse Search
        X_coarse = np.zeros((n_coarse, self.dim))
        x1_coarse = np.linspace(lo[0], hi[0], n_coarse)
        X_coarse[:, 0] = x1_coarse
        X_coarse[:, 1:] = lo[1:]
        
        F_coarse = np.array([problem(x) for x in X_coarse])
        
        # Check time
        if time.time() > deadline:
            return None

        # 2. Find Top K candidates, ensuring spread
        # Divide x1 range into K bins
        candidates = []
        bin_edges = np.linspace(lo[0], hi[0], K + 1)
        
        for i in range(K):
            mask = (x1_coarse >= bin_edges[i]) & (x1_coarse < bin_edges[i+1])
            if i == K - 1:
                mask = (x1_coarse >= bin_edges[i]) & (x1_coarse <= bin_edges[i+1])
            
            if np.any(mask):
                indices_in_bin = np.where(mask)[0]
                # Find index with min f2 in this bin
                best_idx_in_bin = indices_in_bin[np.argmin(F_coarse[indices_in_bin, 1])]
                candidates.append(best_idx_in_bin)
            else:
                # Fallback if bin is empty (shouldn't happen with linspace)
                candidates.append(0)
        
        X_fine = []
        F_fine = []
        
        for idx in candidates:
            x_center = X_coarse[idx, 0]
            
            # Define a small interval around x_center
            # Use the coarse grid step as the half-width
            delta = (hi[0] - lo[0]) / n_coarse
            
            # Sample n_fine_per_candidate points in [x_center - delta, x_center + delta]
            # Clip to bounds
            x_low = max(lo[0], x_center - delta)
            x_high = min(hi[0], x_center + delta)
            
            if x_high <= x_low:
                continue
                
            x1_fine = np.linspace(x_low, x_high, n_fine_per_candidate)
            
            for x1 in x1_fine:
                if time.time() > deadline:
                    return None
                    
                x_trial = np.zeros(self.dim)
                x_trial[0] = x1
                x_trial[1:] = lo[1:]
                
                f_trial = problem(x_trial)
                X_fine.append(x_trial)
                F_fine.append(f_trial)
                
                # Check budget
                if len(F_coarse) + len(F_fine) >= n_evals:
                    break
            if len(F_coarse) + len(F_fine) >= n_evals:
                break
                
        return None
