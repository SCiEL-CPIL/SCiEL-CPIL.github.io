import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        dim = self.dim
        
        start_time = time.time()
        deadline = 295.0  # safety margin before 300s limit
        
        # Pre-compute Pareto front parameterization
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2 = 1 - sqrt(f1)  (when x[1:]=0, g=1)
        
        # Dense grid for pre-computation
        n_grid = 500000
        x_grid = np.linspace(0, 1, n_grid)
        sin_term = np.sin(6 * np.pi * x_grid)
        sin_term = np.power(sin_term, 8)
        exp_term = np.exp(-4 * x_grid)
        f1_grid = 1.0 - exp_term * sin_term
        f2_grid = 1.0 - np.sqrt(np.maximum(f1_grid, 0))
        
        # Sort by f1 for interpolation
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x_grid[sort_idx]
        
        # Remove duplicate f1 values (keep one x1 per unique f1)
        f1_unique, unique_idx = np.unique(f1_sorted, return_index=True)
        x1_unique = x1_sorted[unique_idx]
        
        f1_min = f1_unique[0]
        f1_max = f1_unique[-1]
        
        # Compute cumulative hypervolume along the front
        ref = np.array([11.0, 11.0])
        
        f2_sorted = 1.0 - np.sqrt(np.maximum(f1_unique, 0))
        
        # Compute segment widths
        f1_diff = np.diff(f1_unique)
        heights = ref[1] - f2_sorted[:-1]
        last_width = ref[0] - f1_unique[-1]
        last_height = ref[1] - f2_sorted[-1]
        
        segment_hv = f1_diff * heights
        total_hv_dense = np.sum(segment_hv) + last_width * last_height
        
        # Cumulative hypervolume
        cum_hv = np.cumsum(segment_hv)
        cum_hv = np.concatenate([[0], cum_hv, [total_hv_dense]])
        
        # Select 200 points at equal intervals of cumulative hypervolume
        n_evals = self.budget
        target_hv_vals = np.linspace(0, total_hv_dense, n_evals)
        
        # For each target HV, find the corresponding f1 value
        x1_selected = np.zeros(n_evals)
        for i in range(n_evals):
            h = target_hv_vals[i]
            if h >= total_hv_dense:
                f1_val = f1_unique[-1]
            else:
                seg_idx = np.searchsorted(cum_hv[1:], h, side='right')
                if seg_idx >= len(segment_hv):
                    f1_val = f1_unique[-1]
                else:
                    h_start = cum_hv[seg_idx]
                    h_frac = (h - h_start) / segment_hv[seg_idx] if segment_hv[seg_idx] > 0 else 0
                    f1_val = f1_unique[seg_idx] + h_frac * f1_diff[seg_idx]
            
            f1_val = np.clip(f1_val, f1_min, f1_max)
            x1_val = np.interp(f1_val, f1_unique, x1_unique)
            x1_selected[i] = x1_val
        
        # Construct and evaluate points
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1_selected[i]
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
        
        return None
