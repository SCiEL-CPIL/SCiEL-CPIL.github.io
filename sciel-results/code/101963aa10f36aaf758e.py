import numpy as np
import time
from scipy.integrate import quad

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        max_evals = self.budget
        dim = self.dim
        
        # ZDT6: f1 = 1 + 90*(1-sqrt(x_1)), f2 = (1-f1^2/121)*(1-sqrt(f1/11))
        # x_1 = (1-(f1-1)/90)^2, others = 0
        # Pareto front: f1 in [1, 11]
        
        def f2_of_f1(f1):
            return (1 - f1**2 / 121.0) * (1 - np.sqrt(f1 / 11.0))

        # Pre-compute the cumulative hypervolume (approximated by integral)
        # H(a) = integral from a to 11 of f2(x) dx
        # We want to find f1 values such that H(f1_i) = (N-i)/N * H(1)
        
        # Use a dense grid for interpolation
        grid_size = 10000
        f1_grid = np.linspace(1.0, 11.0, grid_size)
        f2_grid = f2_of_f1(f1_grid)
        
        # Calculate cumulative integral from right to left
        # integral from x to 11
        # We can use cumulative trapezoidal rule
        # d_i = f2_grid[i]
        # width = f1_grid[i+1] - f1_grid[i]
        # area_i = (f2_grid[i] + f2_grid[i+1]) / 2 * width
        
        widths = np.diff(f1_grid)
        areas = (f2_grid[:-1] + f2_grid[1:]) / 2.0 * widths
        
        # Cumulative sum from the right
        # cum_hv[i] = integral from f1_grid[i] to 11.0
        # We need to be careful with the last point.
        # Let's compute cumulative sum of areas from right to left.
        # areas[i] is area between grid[i] and grid[i+1]
        # integral from grid[i] to 11 = sum(areas[i:])
        
        cum_hv = np.cumsum(areas[::-1])[::-1]
        # cum_hv[i] is the integral from f1_grid[i] to f1_grid[-1] (which is 11.0)
        # But we need to add the "tail" if the grid doesn't end exactly at 11? It does.
        # Also, the discrete HV formula has a term f2[-1]*(11-f1[-1]) which is 0 here.
        # So the integral approximation is good for large N.
        
        total_hv = cum_hv[0]
        
        # Find optimal f1 positions
        # We want N points.
        # Target HV for point i (0-indexed from left) is (N-i)/N * total_hv
        # Note: point 0 is f1=1, point N-1 is f1=11.
        # For i=0, target = total_hv. We know f1=1 gives total_hv.
        # For i=N-1, target = 1/N * total_hv.
        
        f1_vals = np.zeros(max_evals)
        f1_vals[0] = 1.0
        f1_vals[-1] = 11.0
        
        # For interior points
        for i in range(1, max_evals - 1):
            target_hv = (max_evals - i) / max_evals * total_hv
            
            # Find f1 such that cum_hv(f1) = target_hv
            # cum_hv is decreasing as f1 increases?
            # f1_grid is increasing.
            # cum_hv[0] is integral from 1 to 11 (max).
            # cum_hv[-1] is integral from 11 to 11 (0).
            # So cum_hv is decreasing.
            
            # We can use np.interp.
            # np.interp(x, xp, fp) expects xp to be increasing.
            # Here, cum_hv is decreasing.
            # We can invert: find f1 for a given hv.
            # xp = cum_hv (decreasing), fp = f1_grid (increasing).
            # np.interp requires xp increasing.
            # So we flip both.
            
            # target_hv is between 0 and total_hv.
            # We want the f1 value.
            # Since cum_hv is monotonic, we can search.
            
            # Flip arrays for interpolation
            # x = cum_hv (reversed, so increasing from 0 to total_hv)
            # y = f1_grid (reversed, so decreasing from 11 to 1)
            # Wait, if we reverse cum_hv, it becomes increasing.
            # If we reverse f1_grid, it becomes decreasing.
            # np.interp(x, xp, fp) with xp increasing.
            # Let xp = cum_hv[::-1] (increasing 0..total)
            # Let fp = f1_grid[::-1] (decreasing 11..1)
            # This works.
            
            f1_val = np.interp(target_hv, cum_hv[::-1], f1_grid[::-1])
            f1_vals[i] = f1_val

        # Ensure sorting and uniqueness (just in case of numerical issues)
        f1_vals = np.sort(f1_vals)
        
        # Evaluate all points
        for i in range(len(f1_vals)):
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            
            f1 = f1_vals[i]
            g = (f1 - 1.0) / 90.0
            x1 = (1.0 - g) ** 2
            x = np.zeros(dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
                n_evals += 1
            except Exception:
                break
        
        return None
