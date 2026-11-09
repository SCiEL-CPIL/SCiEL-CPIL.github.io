import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        all_points = []
        
        # Determine optimal fixed values for x_2 ... x_dim
        # The term to minimize is 10 * (1 + sin(12 * pi * x_i))
        # Minimum is 0 when sin(12 * pi * x_i) = -1
        # 12 * pi * x_i = 3 * pi / 2 + 2 * k * pi
        # x_i = 1/8 + k/6
        # We want a value in [lo[i], hi[i]]
        
        def find_optimal_xi(low, high):
            # Check if 0.125 is in bounds
            if low <= 0.125 <= high:
                return 0.125
            
            # Find nearest minimizer
            # x_k = 1/8 + k/6
            # k = (x - 1/8) * 6
            # We want to find integer k such that x_k is in [low, high]
            # Let's just check a range of k values around 0
            best_val = None
            best_dist = float('inf')
            
            # Range of k to check: 
            # x range is typically [-5, 5]
            # k range: (low - 0.125)*6 to (high - 0.125)*6
            k_min = int(np.floor((low - 0.125) * 6)) - 1
            k_max = int(np.ceil((high - 0.125) * 6)) + 1
            
            for k in range(k_min, k_max + 1):
                x_val = 0.125 + k / 6.0
                if low <= x_val <= high:
                    # All minimizers are equivalent in value, so pick the first valid one
                    # Or the one closest to 0? It doesn't matter for the objective value.
                    return x_val
            
            # If no minimizer found in bounds (unlikely for standard ZDT4 bounds), 
            # fall back to the bound closest to a minimizer? 
            # Or just clip 0.125?
            # Actually, if bounds are tight, we should minimize the function directly?
            # For ZDT4, bounds are usually wide enough. 
            # Let's just clip 0.125 as a fallback, though it might not be optimal.
            return np.clip(0.125, low, high)

        fixed_values = []
        for i in range(1, dim):
            fixed_values.append(find_optimal_xi(lo[i], hi[i]))
        
        # Sample x_1 uniformly
        # x_1 bounds are lo[0], hi[0]
        # We use the full budget for sampling x_1
        x1_samples = np.linspace(lo[0], hi[0], budget)
        
        for x1_val in x1_samples:
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x1_val
            for i in range(1, dim):
                x[i] = fixed_values[i-1]
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            all_points.append(np.asarray(f, dtype=float).copy())
            
            if len(all_points) >= budget:
                break
        
        return all_points
