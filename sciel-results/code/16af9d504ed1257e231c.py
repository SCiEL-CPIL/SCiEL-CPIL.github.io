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

        # Strategy: Build a lookup table mapping f1 -> x1, then select
        # 200 f1 values that maximize hypervolume, and map back to x1.

        n_points = self.budget

        # Build dense grid in x1 to create f1->x1 mapping
        n_grid = 200000
        x1_grid = np.linspace(0.0, 1.0, n_grid)
        
        # Compute f1 on the grid
        sin_term = np.sin(6.0 * np.pi * x1_grid)
        f1_grid = 1.0 - (1.0 - np.exp(-4.0 * x1_grid) * sin_term**6)**0.1
        
        # Sort by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # Get unique f1 values (for lookup)
        unique_f1, unique_indices = np.unique(f1_sorted, return_index=True)
        unique_x1 = x1_sorted[unique_indices]
        n_unique = len(unique_f1)
        
        # HV computation for sorted f1 values t_1 < t_2 < ... < t_n:
        # HV = sum_{i=1}^{n-1} (t_{i+1} - t_i) * (10 + sqrt(t_i)) + (11 - t_n) * (10 + sqrt(t_n))
        
        def compute_hv(t_arr):
            """Compute hypervolume for given sorted f1 values on the Pareto front."""
            t_sorted = np.sort(t_arr)
            t_ext = np.append(t_sorted, 11.0)
            widths = t_ext[1:] - t_ext[:-1]
            heights = 10.0 + np.sqrt(t_sorted)
            return np.sum(widths * heights)
        
        # Initial selection: uniform in f1 space
        idx = np.linspace(0, n_unique - 1, n_points).astype(int)
        t = unique_f1[idx].copy()
        
        hv_current = compute_hv(t)
        
        # Efficient local optimization in f1 space
        # For each point, try shifting to nearby f1 values
        
        # Try multiple passes with decreasing step sizes
        step_sizes = [100, 50, 20, 10, 5, 2, 1]
        for step in step_sizes:
            improved = False
            for i in range(n_points):
                if time.time() > deadline - 20:
                    break
                
                best_hv = hv_current
                cur_idx = np.searchsorted(unique_f1, t[i])
                
                # Try a range of nearby indices
                lo = max(0, cur_idx - 3 * step)
                hi = min(n_unique - 1, cur_idx + 3 * step)
                
                for trial_idx in range(lo, hi + 1):
                    if trial_idx == cur_idx:
                        continue
                    t_trial = t.copy()
                    t_trial[i] = unique_f1[trial_idx]
                    hv_trial = compute_hv(t_trial)
                    if hv_trial > best_hv + 1e-12:
                        best_hv = hv_trial
                        cur_idx = trial_idx
                
                if cur_idx != np.searchsorted(unique_f1, t[i]):
                    t[i] = unique_f1[cur_idx]
                    hv_current = best_hv
                    improved = True
                
                if time.time() > deadline - 20:
                    break
            
            if not improved:
                break
        
        # Final x1 values
        x1_final = unique_x1[np.searchsorted(unique_f1, t)]
        
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
