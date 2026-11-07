import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Initialize bounds
        bounds_low = np.full(dim, -5.0)
        bounds_high = np.full(dim, 5.0)
        
        global_best_x = np.full(dim, 0.0)
        global_best_fx = np.inf
        
        # Phase 1: Recursive 2^5 Grid Subdivision
        # We subdivide the current best cell into 2^5 sub-cells and evaluate their centers.
        while budget >= 32:
            centers = []
            cell_bounds = []
            
            # Iterate through all 2^5 sub-cells
            for mask in range(32):
                c = np.zeros(dim)
                b_low = np.zeros(dim)
                b_high = np.zeros(dim)
                
                for i in range(dim):
                    mid = (bounds_low[i] + bounds_high[i]) / 2.0
                    if (mask >> i) & 1:
                        # Upper half
                        b_low[i] = mid
                        b_high[i] = bounds_high[i]
                        c[i] = (mid + bounds_high[i]) / 2.0
                    else:
                        # Lower half
                        b_low[i] = bounds_low[i]
                        b_high[i] = mid
                        c[i] = (bounds_low[i] + mid) / 2.0
                        
                centers.append(c)
                cell_bounds.append((b_low, b_high))
                
            centers = np.array(centers)
            f_vals = np.array([func(x) for x in centers])
            budget -= len(centers)
            
            best_idx = np.argmin(f_vals)
            best_val = f_vals[best_idx]
            
            if best_val < global_best_fx:
                global_best_fx = best_val
                global_best_x = centers[best_idx].copy()
                
            # Update search space to the best sub-cell
            bounds_low, bounds_high = cell_bounds[best_idx]
            
        # Phase 2: Local Polish with remaining budget
        # If we have budget left, perform a fine local search around global_best_x
        if budget > 0:
            # Use a small grid around the best point
            # Estimate step size based on current cell width
            current_width = (bounds_high - bounds_low)
            step = np.max(current_width) / 2.0
            
            # Generate a small 3^dim grid around the best point?
            # Or just use the remaining budget for a (1+1) ES with small sigma
            rng = np.random.default_rng(42)
            x = global_best_x.copy()
            fx = global_best_fx
            sigma = step * 0.5
            
            for _ in range(budget):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    
                if fx < global_best_fx:
                    global_best_fx = fx
                    global_best_x = x.copy()

        return global_best_x, global_best_fx
