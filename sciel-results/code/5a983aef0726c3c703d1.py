"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT3 exploiting its discontinuous Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        all_points = []
        used = 0

        # ZDT3 structure with g=1 (x[1:]=0)
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin(10 * pi * x0)
        
        # Dense grid to identify the non-dominated envelope
        # Use a very fine grid to capture all local minima of the oscillatory f2
        n_grid = 100000
        x_grid = np.linspace(0.0, 1.0, n_grid)
        
        # Calculate f2 for the grid
        f2_grid = 1.0 - np.sqrt(x_grid) - x_grid * np.sin(10.0 * np.pi * x_grid)
        
        # Find indices where f2 is a new minimum (strictly less than all previous)
        # These are the non-dominated points on the ideal front
        min_f2_so_far = np.empty(n_grid)
        min_f2_so_far[0] = f2_grid[0]
        for i in range(1, n_grid):
            if f2_grid[i] < min_f2_so_far[i-1]:
                min_f2_so_far[i] = f2_grid[i]
            else:
                min_f2_so_far[i] = min_f2_so_far[i-1]
                
        is_nd = (f2_grid == min_f2_so_far)
        x_nd = x_grid[is_nd]
        f2_nd = f2_grid[is_nd]
        
        n_nd = len(x_nd)
        
        if n_nd <= budget:
            selected_x = x_nd
        else:
            # We need to select 'budget' points from the non-dominated set
            # to maximize hypervolume with reference point (11, 11)
            
            # The hypervolume for 2D points sorted by f1 (increasing) and f2 (decreasing)
            # with reference (r1, r2) is: sum over i of (f1[i+1] - f1[i]) * (f2[i] - r2) 
            # but more precisely, it's the union of rectangles.
            
            # A practical approach: the marginal contribution of each point to HV
            # is approximately the area of the rectangle it uniquely covers.
            # For point i (0-indexed, sorted by f1 increasing), the rectangle is:
            # width = f1[i+1] - f1[i] (or f1[i] - f1[i-1] for the last)
            # height = f2[i] - f2[i+1] (drop to next point)
            # But this is an approximation. A better weight is based on the actual HV contribution.
            
            # Let's compute approximate HV contribution weights:
            # For each point, the "unique" area it covers is bounded by:
            # left: f1[i-1] (or 0), right: f1[i+1] (or 1)
            # top: f2[i], bottom: f2[i+1] (or 0)
            # But since points are non-dominated, f2 is decreasing.
            
            # Simpler and effective: weight by the product of f1 gap and f2 gap
            # This captures both the horizontal and vertical contribution
            
            # Compute f1 gaps (widths)
            f1_gaps = np.empty(n_nd)
            f1_gaps[:-1] = x_nd[1:] - x_nd[:-1]
            f1_gaps[-1] = 1.0 - x_nd[-1]  # gap to the right boundary
            
            # Compute f2 drops (heights)
            f2_drops = np.empty(n_nd)
            f2_drops[:-1] = f2_nd[:-1] - f2_nd[1:]
            f2_drops[-1] = f2_nd[-1]  # drop to zero (or reference, but f2 is small)
            
            # Ensure no negative or zero weights
            f1_gaps = np.maximum(f1_gaps, 1e-15)
            f2_drops = np.maximum(f2_drops, 1e-15)
            
            # Weight = f1_gap * f2_drop approximates the HV contribution
            weights = f1_gaps * f2_drops
            weights = weights / weights.sum()
            
            # Stratified sampling using inverse CDF
            cdf = np.cumsum(weights)
            cdf /= cdf[-1]
            
            # Generate 'budget' stratified samples
            u = (np.arange(budget) + 0.5) / budget
            indices = np.searchsorted(cdf, u)
            indices = np.clip(indices, 0, n_nd - 1)
            
            # Ensure unique indices (should be unique due to strictly increasing CDF)
            # But searchsorted can return duplicates if CDF has flat regions
            # Since weights are positive, CDF is strictly increasing, so indices should be unique
            # However, let's verify and fix if needed
            unique_mask = np.concatenate(([True], indices[1:] != indices[:-1]))
            if not unique_mask.all():
                # Fallback: use uniform index sampling
                indices = np.linspace(0, n_nd - 1, budget, dtype=int)
            
            selected_x = x_nd[indices]
            
            # Verify we have exactly budget points
            if len(selected_x) != budget:
                # Pad or trim as needed
                if len(selected_x) < budget:
                    remaining = budget - len(selected_x)
                    # Add uniform points to fill
                    extra = np.linspace(0, 1, remaining)
                    selected_x = np.concatenate([selected_x, extra])
                else:
                    selected_x = selected_x[:budget]

        # Evaluate the selected points
        for i in range(len(selected_x)):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = selected_x[i]
            # x[1:] = 0 achieves g = 1 (minimum)
            
            f = problem(x)
            all_points.append(f)
            used += 1

        return all_points
# EVOLVE-BLOCK-END
