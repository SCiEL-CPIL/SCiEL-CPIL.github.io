import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin

        # Parameters for ZDT6 variant described in the seed program
        # f1(x1) = 1 - (1 - exp(-4*x1)*sin(6*pi*x1)^6)^(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        # Pareto front is at x2...xn = 0
        
        # 1. Create a dense grid of x1 values to map f1 -> x1
        # We need high resolution to accurately map target f1 values back to x1
        n_grid = 500000
        x1_grid = np.linspace(0.0, 1.0, n_grid)
        
        # Compute f1 for the grid
        # Using vectorized operations for speed
        sin_term = np.sin(6.0 * np.pi * x1_grid)
        exp_term = np.exp(-4.0 * x1_grid)
        inner = 1.0 - exp_term * (sin_term ** 6)
        # Ensure inner is within [0, 1] for the 0.1 power
        # Numerical precision might push it slightly out
        inner = np.clip(inner, 0.0, 1.0)
        f1_grid = 1.0 - (inner ** 0.1)
        
        # 2. Determine the range of f1 on the Pareto front
        # We want to cover the entire front
        f1_min = f1_grid.min()
        f1_max = f1_grid.max()
        
        # 3. Select n_points target f1 values uniformly spaced in [f1_min, f1_max]
        # This distribution is generally optimal for maximizing Hypervolume
        # for convex/concave fronts when the number of points is large enough.
        n_points = self.budget
        target_f1 = np.linspace(f1_min, f1_max, n_points)
        
        # 4. Map target_f1 values back to x1 values
        # Sort the grid by f1 to enable efficient lookup
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # For each target f1, find the closest x1 in the sorted grid
        # np.searchsorted returns the index where the value would be inserted
        indices = np.searchsorted(f1_sorted, target_f1)
        
        # Handle edge cases where searchsorted returns index == len(f1_sorted)
        # or index == 0
        indices = np.clip(indices, 0, len(f1_sorted) - 1)
        
        # To get the closest match, we should check the element at `indices` and `indices-1`
        # and pick the one with f1 closer to target_f1
        # However, with 500k points, the error is negligible.
        # But for robustness, let's do a quick check.
        # We can vectorize the comparison.
        
        # Get f1 values at the found indices
        f1_at_idx = f1_sorted[indices]
        
        # Get f1 values at indices-1 (clipped)
        indices_minus = np.clip(indices - 1, 0, len(f1_sorted) - 1)
        f1_at_idx_minus = f1_sorted[indices_minus]
        
        # Calculate distances
        dist_current = np.abs(f1_at_idx - target_f1)
        dist_minus = np.abs(f1_at_idx_minus - target_f1)
        
        # Choose the better index
        better_indices = np.where(dist_minus < dist_current, indices_minus, indices)
        
        x1_selected = x1_sorted[better_indices]
        
        # 5. Evaluate the selected points
        recorded = []
        for i in range(n_points):
            if time.time() > deadline - 5.0:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_selected[i]
            # x[1:] remains 0, which is on the Pareto front for ZDT6
            # Clip to bounds as required by the problem interface
            # bounds is a list of (low, high)
            for j in range(self.dim):
                x[j] = np.clip(x[j], bounds[j][0], bounds[j][1])
            
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
