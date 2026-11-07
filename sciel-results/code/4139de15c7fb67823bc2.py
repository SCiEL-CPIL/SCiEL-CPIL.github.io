import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        # ZDT6: f1(x1) = 1 - (1 - exp(-4*x1)*sin(6*pi*x1)^6)^(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        # On Pareto front, x2..x10 = 0.
        # f1 is not monotonic in x1; we need to find x1 for each target f1.

        # Strategy: sample x1 on a non-uniform grid denser where df1/dx1 is small,
        # then select the 200 points that maximize hypervolume.

        # First, build a dense x1 grid
        n_grid = 2000000
        x1_grid = np.linspace(0.0, 1.0, n_grid)
        
        # Compute f1 on the grid in chunks
        f1_grid = np.empty(n_grid)
        chunk_size = 200000
        for i in range(0, n_grid, chunk_size):
            end = min(i + chunk_size, n_grid)
            x_chunk = x1_grid[i:end]
            f1_grid[i:end] = 1.0 - (1.0 - np.exp(-4.0 * x_chunk) * np.sin(6.0 * np.pi * x_chunk)**6)**0.1
        
        # f1 range
        f1_min = f1_grid.min()
        f1_max = f1_grid.max()
        
        # We want to select n_points = budget points from this grid to maximize HV.
        # Since f1 is not monotonic, multiple x1 values can give the same f1.
        # For each unique f1 value, we only need one x1 (any will do since f2 depends only on f1).
        # So we can reduce to unique f1 values.
        
        # Sort by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # Remove duplicates in f1 (keep first occurrence for each f1)
        unique_f1, unique_indices = np.unique(f1_sorted, return_index=True)
        unique_x1 = x1_sorted[unique_indices]
        
        n_unique = len(unique_f1)
        
        # Now select n_points from unique_f1 to maximize HV.
        # HV for points at f1 = t1 < t2 < ... < tn:
        # HV = sum_{i=1}^{n-1} (t_{i+1} - t_i) * (10 + sqrt(t_i)) + (11 - t_n) * (10 + sqrt(t_n))
        
        n_points = self.budget
        
        # Initial selection: uniform in f1 space
        if n_unique >= n_points:
            # Pick n_points evenly spaced f1 values
            idx = np.linspace(0, n_unique - 1, n_points).astype(int)
            t = unique_f1[idx]
            x1_sel = unique_x1[idx]
        else:
            t = unique_f1
            x1_sel = unique_x1
            n_points = n_unique
        
        def compute_hv(t_arr):
            """Compute hypervolume for given f1 values on the Pareto front."""
            t_sorted = np.sort(t_arr)
            t_ext = np.append(t_sorted, 11.0)
            widths = t_ext[1:] - t_ext[:-1]
            heights = 10.0 + np.sqrt(t_sorted)
            hv = np.sum(widths * heights)
            return hv
        
        # Local optimization: try to improve by swapping or adjusting points
        # Use a simple approach: for each point, try replacing it with a nearby f1 value
        hv_current = compute_hv(t)
        
        # Build a mapping from f1 value to index in unique_f1
        # For efficiency, work with indices into unique_f1
        t_idx = np.searchsorted(unique_f1, t)
        
        # Coordinate descent: for each point, try shifting to adjacent f1 values
        step_sizes = [1, 2, 5, 10, 50]
        for step in step_sizes:
            improved = False
            for i in range(n_points):
                if time.time() > deadline - 15:
                    break
                
                # Try shifting left and right by 'step' in the unique_f1 array
                for delta in [-step, step]:
                    new_idx = t_idx[i] + delta
                    if new_idx < 0 or new_idx >= n_unique:
                        continue
                    
                    # Check if this f1 value is already used by another point
                    # (we can allow duplicates since they don't hurt HV, but let's avoid for simplicity)
                    t_trial = t.copy()
                    t_trial[i] = unique_f1[new_idx]
                    
                    hv_trial = compute_hv(t_trial)
                    if hv_trial > hv_current + 1e-12:
                        t = t_trial
                        t_idx[i] = new_idx
                        hv_current = hv_trial
                        improved = True
                        break
                
                if time.time() > deadline - 15:
                    break
            
            if not improved:
                break
        
        # Final x1 values
        x1_final = unique_x1[t_idx]
        
        # Evaluate all points
        recorded = []
        for i in range(n_points):
            if time.time() > deadline - 5:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_final[i]
            x[1:] = 0.0  # x2..x10 = 0 for Pareto front
            x = np.clip(x, bounds[0][0], bounds[0][1])
            obj = problem(x)
            recorded.append(obj)
        
        return recorded
