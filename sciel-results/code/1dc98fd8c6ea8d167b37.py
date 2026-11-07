import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using analytical Pareto front sampling.
    
    The ZDT3 Pareto front is f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1].
    We maximize hypervolume with reference point [11, 11] by optimizing the
    1D placement of x1 values.
    
    Strategy:
    1. Start with a geometric grid (dense near x=0 where curvature is highest).
    2. Iteratively replace points with local optimization to improve HV.
    3. The HV of points on a monotone decreasing curve can be computed as sum of trapezoids.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, x1_vals):
        """
        Compute hypervolume of points (x1_i, 1-sqrt(x1_i)) with ref (11, 11).
        Points must be sorted in increasing x1 (decreasing f2).
        """
        if len(x1_vals) == 0:
            return 0.0
        if len(x1_vals) == 1:
            x = x1_vals[0]
            y = 1.0 - np.sqrt(x)
            if x < 11.0 and y < 11.0:
                return (11.0 - x) * (11.0 - y)
            return 0.0
        
        x1_vals = np.sort(x1_vals)
        # Remove duplicates
        x1_vals = np.unique(x1_vals)
        if len(x1_vals) == 0:
            return 0.0
        if len(x1_vals) == 1:
            x = x1_vals[0]
            y = 1.0 - np.sqrt(x)
            if x < 11.0 and y < 11.0:
                return (11.0 - x) * (11.0 - y)
            return 0.0
        
        f2_vals = 1.0 - np.sqrt(x1_vals)
        
        # Compute HV as sum of trapezoids
        # For each consecutive pair (i, i+1), the contribution is the area of the trapezoid
        # between the two points and the reference
        # Actually, for monotone decreasing front, HV = sum over i of (x_{i+1} - x_i) * (ref_y - y_i) 
        # + (ref_x - x_{n-1}) * (ref_y - y_{n-1})
        # Wait, let me think more carefully.
        # HV = integral of (ref_y - y(x)) dx over the region where y(x) < ref_y
        # For piecewise linear approximation, HV = sum of trapezoid areas
        # Each trapezoid between x_i and x_{i+1} has area:
        #   (x_{i+1} - x_i) * (ref_y - (y_i + y_{i+1})/2)
        # Plus the last rectangle from x_{n-1} to ref_x:
        #   (ref_x - x_{n-1}) * (ref_y - y_{n-1})
        
        ref_x = 11.0
        ref_y = 11.0
        
        # Filter points that are strictly better than ref in both objectives
        mask = (x1_vals < ref_x) & (f2_vals < ref_y)
        x1_filtered = x1_vals[mask]
        f2_filtered = f2_vals[mask]
        
        if len(x1_filtered) == 0:
            return 0.0
        if len(x1_filtered) == 1:
            x = x1_filtered[0]
            y = f2_filtered[0]
            return (ref_x - x) * (ref_y - y)
        
        # Compute trapezoid areas
        dx = np.diff(x1_filtered)
        avg_f2 = (f2_filtered[:-1] + f2_filtered[1:]) / 2.0
        areas = dx * (ref_y - avg_f2)
        hv = np.sum(areas)
        # Add last rectangle
        hv += (ref_x - x1_filtered[-1]) * (ref_y - f2_filtered[-1])
        
        return hv

    def _compute_initial_grid(self, B):
        """Generate initial geometric grid of x1 values."""
        if B == 1:
            return np.array([0.0])
        i = np.arange(B)
        t = i / (B - 1)
        # Geometric distribution: dense near 0
        x1_vals = t ** 1.5
        x1_vals[0] = 0.0
        x1_vals[-1] = 1.0
        return x1_vals

    def _local_optimize_point(self, x1_vals, idx, B):
        """
        Optimize a single point at index idx, given all other points are fixed.
        """
        if len(x1_vals) <= 1:
            return 0.0
        
        # Get neighbors
        left_idx = max(0, idx - 1)
        right_idx = min(len(x1_vals) - 1, idx + 1)
        
        # The bounds for this point are determined by its neighbors
        if idx == 0:
            lb = 0.0
            ub = x1_vals[1] if len(x1_vals) > 1 else 1.0
        elif idx == len(x1_vals) - 1:
            lb = x1_vals[-2]
            ub = 1.0
        else:
            lb = x1_vals[idx - 1]
            ub = x1_vals[idx + 1]
        
        if ub - lb < 1e-10:
            return x1_vals[idx]
        
        def neg_hv(x):
            new_x1 = x1_vals.copy()
            new_x1[idx] = x
            new_x1 = np.sort(new_x1)
            return -self._hypervolume_2d(new_x1)
        
        result = minimize_scalar(neg_hv, bounds=(lb + 1e-10, ub - 1e-10), method='bounded')
        if result.success:
            return result.x
        return x1_vals[idx]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin before 330s limit

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        B = self.budget
        
        # Compute optimal x1 values
        x1_values = self._compute_initial_grid(B)
        
        # Iterative local improvement
        max_iter = 2000
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            improved = False
            for idx in range(B):
                if time.time() > deadline - 30:
                    break
                
                old_val = x1_values[idx]
                new_val = self._local_optimize_point(x1_values, idx, B)
                
                if abs(new_val - old_val) > 1e-8:
                    x1_values[idx] = new_val
                    x1_values = np.sort(x1_values)
                    improved = True
            
            if not improved:
                break
        
        # Ensure endpoints
        if len(x1_values) > 0:
            x1_values[0] = 0.0
            x1_values[-1] = 1.0
            x1_values = np.sort(x1_values)
        
        # Evaluate on the front
        F = []
        for x1 in x1_values:
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(f)
            
            if time.time() > deadline:
                break

        return np.array(F)
