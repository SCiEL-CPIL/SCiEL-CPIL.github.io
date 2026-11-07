import numpy as np
import time

class Optimizer:
    """
    Optimizer using a coarse-to-fine grid search followed by local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls = 0
        max_calls = self.budget
        dim = self.dim
        bound = self.bound

        best_x = None
        best_f = float('inf')

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls or time.time() > deadline:
                return float('inf')
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse-to-Fine Grid Search
        # Start with a coarse grid and progressively refine the best region
        # Level 0: 2^5 = 32 points (too many for 5D)
        # Let's use a more structured approach: divide into 3^dim cells per level
        
        # For 5D, 3^5 = 243 cells per level is manageable for a few levels
        # But with 500 budget, we need to be careful.
        # Level 0: 3^5 = 243 calls
        # Level 1: 3^5 = 243 calls on the best sub-region
        # Total: 486, leaving 14 for refinement
        
        # Actually, let's be more strategic. 
        # Level 0: 2^5 = 32 points (corners of 3^5 grid minus corners? No, centers of 2^5 cells)
        # Let's use 3 divisions per axis at level 0: 3^5 = 243
        # Then take the best 1 cell and subdivide it into 3^5 = 243 more
        # Total: 486, leaving 14 for local refinement
        
        # But wait, the budget is 500. 243 + 243 = 486. That leaves only 14.
        # Let's do:
        # Level 0: 2^5 = 32 points
        # Level 1: 3^5 = 243 points on best 1/32 region
        # Level 2: 3^5 = 243 points on best 1/243 region of that
        # Total: 32 + 243 + 243 = 518 > 500. Too many.
        
        # Revised plan:
        # Level 0: 2^5 = 32 points
        # Level 1: 3^5 = 243 points on best region
        # Total: 275, leaving 225 for more refinement or local ES
        
        # Actually, let's do a single coarse grid of 3^5 = 243, then subdivide the best cell into 3^5 = 243.
        # That's 486 calls. Then 14 calls for local refinement.
        
        def get_grid_centers(low, high, n_div):
            """Generate centers of n_div^dim cells in the hypercube [low, high]^dim"""
            l = np.array(low)
            h = np.array(high)
            step = (h - l) / n_div
            centers = []
            # Use recursive product
            indices = np.indices([n_div] * dim)
            for idx in indices.reshape(-1, dim):
                center = l + (idx + 0.5) * step
                centers.append(center)
            return np.array(centers)
        
        # Level 0: 3^5 = 243 points over the full domain
        if calls < max_calls and time.time() < deadline:
            grid0 = get_grid_centers(-bound, bound, 3)
            for x in grid0:
                if time.time() > deadline:
                    break
                if calls >= max_calls:
                    break
                f = eval_func(x)
                if not np.isfinite(f):
                    break
            
            if calls >= max_calls or time.time() > deadline:
                return best_x, best_f
        
        # Find the best cell from level 0
        # We need to track which cell each point belongs to
        # Redo with tracking
        calls_so_far = calls
        if calls_so_far < max_calls and time.time() < deadline:
            # We already evaluated grid0, but we didn't track which was best cell
            # Let's just use best_x and find the cell it's in
            # Actually, for simplicity, let's just subdivide the region around best_x
            
            # Determine which cell best_x is in
            cell_size = 2 * bound / 3
            cell_idx = np.floor((best_x + bound) / cell_size).astype(int)
            cell_idx = np.clip(cell_idx, 0, 2)
            
            # The best cell is [cell_idx * cell_size - bound, (cell_idx + 1) * cell_size - bound]
            cell_low = cell_idx * cell_size - bound
            cell_high = (cell_idx + 1) * cell_size - bound
            
            # Level 1: 3^5 = 243 points in the best cell
            if calls < max_calls and time.time() < deadline:
                grid1 = get_grid_centers(cell_low, cell_high, 3)
                for x in grid1:
                    if time.time() > deadline:
                        break
                    if calls >= max_calls:
                        break
                    f = eval_func(x)
                    if not np.isfinite(f):
                        break
            
            if calls >= max_calls or time.time() > deadline:
                return best_x, best_f
            
            # Find the best sub-cell from level 1
            cell_size1 = cell_size / 3
            cell_idx1 = np.floor((best_x - cell_low) / cell_size1).astype(int)
            cell_idx1 = np.clip(cell_idx1, 0, 2)
            
            sub_cell_low = cell_low + cell_idx1 * cell_size1
            sub_cell_high = cell_low + (cell_idx1 + 1) * cell_size1
            
            # Level 2: Use remaining budget for finer grid or local ES
            remaining = max_calls - calls
            if remaining > 0 and time.time() < deadline:
                # If we have enough budget, do another level of grid
                if remaining >= 243:
                    grid2 = get_grid_centers(sub_cell_low, sub_cell_high, 3)
                    for x in grid2:
                        if time.time() > deadline:
                            break
                        if calls >= max_calls:
                            break
                        f = eval_func(x)
                        if not np.isfinite(f):
                            break
                else:
                    # Use remaining budget for local refinement with (1+1)-ES
                    rng = np.random.default_rng(42)
                    x = best_x.copy()
                    fx = best_f
                    sigma = cell_size1 * 0.3
                    
                    for _ in range(remaining):
                        if time.time() > deadline:
                            break
                        if calls >= max_calls:
                            break
                        y = np.clip(x + sigma * rng.standard_normal(dim), -bound, bound)
                        fy = eval_func(y)
                        if not np.isfinite(fy):
                            break
                        if fy < fx:
                            x, fx = y, fy
                            sigma *= 1.2
                        else:
                            sigma *= 0.8
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
        
        # Final local refinement with remaining budget
        remaining = max_calls - calls
        if remaining > 0 and time.time() < deadline and best_x is not None:
            rng = np.random.default_rng(123)
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls >= max_calls:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -bound, bound)
                fy = eval_func(y)
                if not np.isfinite(fy):
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)
        
        return best_x, best_f
