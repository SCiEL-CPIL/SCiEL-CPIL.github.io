"""Deterministic hierarchical grid search for Katsuura (f23)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hierarchical grid-based optimizer exploiting Katsuura's structure."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds_lo = -5.0
        bounds_hi = 5.0

        # We'll track the best point and value found so far
        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return best_f
            calls_used += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid with 3 points per dimension (3^5 = 243 evaluations)
        # This gives us a broad overview of the landscape
        grid_sizes = [3, 5, 7]
        current_lo = np.array([bounds_lo] * dim)
        current_hi = np.array([bounds_hi] * dim)

        # First pass: coarse 3^5 grid
        for size in grid_sizes:
            if calls_used >= budget:
                break
            # Generate grid points
            coords = np.linspace(current_lo, current_hi, size)
            # Use itertools.product to generate all combinations
            from itertools import product
            grid_points = list(product(*[coords] * dim))
            
            # For large grids, we need to be careful about budget
            # 3^5=243, 5^5=3125, 7^5=16807
            # We can only afford the first one fully, so let's be smarter
            
            if len(grid_points) <= budget - calls_used:
                # Evaluate all points
                for pt in grid_points:
                    x = np.array(pt)
                    evaluate(x)
                    if calls_used >= budget:
                        break
            else:
                # Can't afford full grid, so just do a random/sample subset
                # Or break and go to next phase with what we have
                break
            
            # After each grid, find the best region and refine
            if best_x is not None:
                # Define a sub-box around the best point for next iteration
                # The box should be roughly the size of one grid cell
                cell_width = (current_hi - current_lo) / (grid_sizes[grid_sizes.index(size) if size in grid_sizes else 0] - 1) if size > 1 else (current_hi - current_lo)
                # Actually, let's compute it properly
                if size > 1:
                    step = (current_hi - current_lo) / (size - 1)
                else:
                    step = current_hi - current_lo
                
                new_lo = np.maximum(bounds_lo, best_x - step)
                new_hi = np.minimum(bounds_hi, best_x + step)
                
                # Only update if the box is valid (non-degenerate)
                if all(new_hi[i] > new_lo[i] for i in range(dim)):
                    current_lo = new_lo
                    current_hi = new_hi
                else:
                    # Box is too small, keep previous
                    break

        # If we still have budget left, do a final local refinement
        # using a fine grid around the best point
        if best_x is not None and calls_used < budget:
            # Determine a small neighborhood
            # Use a fraction of the current box size
            if 'current_lo' in dir() and 'current_hi' in dir():
                box_size = current_hi - current_lo
            else:
                box_size = np.array([10.0] * dim)
            
            # Try a few finer grids with increasing resolution
            for fine_size in [5, 7, 9]:
                if calls_used >= budget:
                    break
                
                # Center on best_x with a small window
                window = box_size * 0.3 / fine_size  # small window
                lo = np.maximum(bounds_lo, best_x - window)
                hi = np.minimum(bounds_hi, best_x + window)
                
                if all(hi[i] > lo[i] for i in range(dim)):
                    coords = np.linspace(lo, hi, fine_size)
                    from itertools import product
                    grid_points = list(product(*[coords] * dim))
                    
                    if len(grid_points) <= budget - calls_used:
                        for pt in grid_points:
                            x = np.array(pt)
                            evaluate(x)
                            if calls_used >= budget:
                                break
                    else:
                        break

        # Final: if we have budget, do a few random perturbations around best
        if best_x is not None and calls_used < budget:
            rng = np.random.default_rng(42)
            # Use remaining budget for local search
            remaining = budget - calls_used
            sigma = 0.1
            for _ in range(remaining):
                if calls_used >= budget:
                    break
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, bounds_lo, bounds_hi)
                fy = func(y)
                calls_used += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
