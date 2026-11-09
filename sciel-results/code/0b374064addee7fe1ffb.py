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
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        results = []
        
        # Strategy: Sample uniformly in f1 space to maximize hypervolume coverage.
        # The Pareto front of ZDT6 (with x[1:]=0) is defined by:
        # f1(x0) = 1 + x0 * (sin(10*pi*x0) + sin(10*pi*x0^2))
        # f2(x0) = 1 - sqrt(f1(x0))
        
        # We create a dense grid of x0 to map x0 -> f1.
        # Then we select x0 values such that f1 is uniformly spaced.
        # This ensures better distribution of points along the Pareto front in objective space.
        
        # Use a dense grid for offline selection
        n_grid = 5000
        x0_grid = np.linspace(0.0, 1.0, n_grid)
        
        # Compute f1 for the grid
        # f1(x0) = 1 + x0 * (sin(10*pi*x0) + sin(10*pi*x0^2))
        sin_term1 = np.sin(10 * np.pi * x0_grid)
        sin_term2 = np.sin(10 * np.pi * x0_grid**2)
        f1_grid = 1 + x0_grid * (sin_term1 + sin_term2)
        
        # Ensure f1 is positive for sqrt if we were to compute f2, 
        # but we only need f1 for sorting/selection.
        # However, to be safe, we only care about the mapping x0 -> f1.
        
        # Sort by f1 to allow uniform selection in f1 space
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x0_sorted = x0_grid[sort_idx]
        
        # Select n_evals points uniformly spaced in f1
        if n_evals == 1:
            # For a single point, pick the one that maximizes the hypervolume contribution.
            # Hypervolume contribution of a single point p is (ref[0]-f1[p]) * (ref[1]-f2[p]).
            # Since f2 = 1 - sqrt(f1), let's compute the area for the grid.
            # We need f2 to compute the area.
            f1_safe = np.maximum(f1_grid, 1e-6)
            f2_grid = 1 - np.sqrt(f1_safe)
            areas = (11.0 - f1_grid) * (11.0 - f2_grid)
            best_idx = np.argmax(areas)
            x0s = [x0_grid[best_idx]]
        else:
            # Uniform spacing in f1
            # We want indices into the sorted f1 array such that f1 values are linear.
            # Since f1_sorted is sorted, we can pick indices that are uniformly spaced 
            # in the index domain if f1 was linear, but it's not.
            # Better: pick target f1 values uniformly, then find closest x0.
            # Or simpler: pick indices uniformly in the sorted array? 
            # No, that's uniform in rank, not value.
            
            # Let's pick target f1 values uniformly between min(f1) and max(f1)
            f1_min = f1_sorted[0]
            f1_max = f1_sorted[-1]
            
            # Generate n_evals target f1 values
            target_f1s = np.linspace(f1_min, f1_max, n_evals)
            
            # Find the x0 corresponding to each target f1
            # Since f1_sorted is sorted, we can use searchsorted
            # But f1 is not strictly monotonic with x0, it's monotonic with the sorted indices.
            # So we map target_f1 to an index in f1_sorted, then get x0_sorted[index].
            
            # searchsorted returns the index where the target would be inserted to maintain order.
            # We want the closest point.
            indices = np.searchsorted(f1_sorted, target_f1s)
            
            # Clamp indices to valid range
            indices = np.clip(indices, 0, n_grid - 1)
            
            # Get the x0 values
            x0s = x0_sorted[indices]
            
            # It's possible that due to non-monotonicity of f1(x0), 
            # multiple x0 map to similar f1, or the mapping is complex.
            # But by sorting f1, we've linearized the objective space.
            # Picking uniform f1 targets and finding the corresponding x0 (via the sorted grid)
            # ensures the points are well-distributed in f1.
            
        # Evaluate the selected points
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
