# EVOLVE-BLOCK-START
import time
import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    num_tubes = N * N
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

    # Pre-compute tube indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(np.float64)
    j_flat = j_grid.flatten().astype(np.float64)
    
    # Order of placement: We place tubes in an order that might help packing.
    # Usually, placing tubes with similar shears (i, j) first might create a dense core.
    # Or placing them in a spiral. Let's try sorting by (i+j) to create diagonal bands.
    order = np.argsort(i_flat + j_flat)
    
    # Fast volume estimator for the search.
    # We use a fixed set of random points in [0,1]^3.
    rng_search = np.random.default_rng(42)
    num_search_pts = 50000
    a_norm = rng_search.uniform(0, 1, num_search_pts)
    b_norm = rng_search.uniform(0, 1, num_search_pts)
    c_samples = rng_search.uniform(0, 1, num_search_pts)
    
    # Candidate positions for x and y.
    # We restrict search to a grid to make it fast and discrete.
    # The box size should be small. Let's assume the box is roughly [0, 1] or smaller.
    # Actually, the optimal box for Kakeya sets is often related to the range of shears.
    # Let's try a grid of positions in [0, 1].
    grid_res = 16
    x_candidates = np.arange(0, 1.0, 1.0/grid_res)
    y_candidates = np.arange(0, 1.0, 1.0/grid_res)
    
    def fast_volume(x, y, subset_idx=None):
        """
        Estimate volume for a subset of tubes defined by x, y, i, j.
        If subset_idx is None, use all.
        """
        if subset_idx is not None:
            x_sub = x[subset_idx]
            y_sub = y[subset_idx]
            i_sub = i_flat[subset_idx]
            j_sub = j_flat[subset_idx]
        else:
            x_sub = x
            y_sub = y
            i_sub = i_flat
            j_sub = j_flat
            
        if len(x_sub) == 0:
            return 0.0

        X0 = np.min(x_sub)
        X1 = np.max(x_sub + (i_sub + 1) / N)
        Y0 = np.min(y_sub)
        Y1 = np.max(y_sub + (j_sub + 1) / N)
        
        w = X1 - X0
        h = Y1 - Y0
        if w <= 1e-9 or h <= 1e-9:
            return 0.0

        # Map normalized samples to the box
        a = X0 + a_norm * w
        b = Y0 + b_norm * h
        
        inside_any = np.zeros(num_search_pts, dtype=bool)
        
        # Process in batches to manage memory, though 64 tubes is small
        # We can just loop over tubes for the search phase as it's called many times
        # but vectorizing over points is key.
        
        # For speed in the greedy step, we only care about the incremental volume.
        # But fast_volume recalculates from scratch.
        # To be faster, we could maintain a coverage array, but that's complex.
        # Let's just use the vectorized check.
        
        for k in range(len(x_sub)):
            bx = x_sub[k]
            by = y_sub[k]
            bi = i_sub[k]
            bj = j_sub[k]
            
            pc = c_samples
            left = bx + pc * bi / N
            right = left + 1.0 / N
            bottom = by + pc * bj / N
            top = bottom + 1.0 / N
            
            in_x = (a >= left) & (a <= right)
            in_y = (b >= bottom) & (b <= top)
            
            inside_any |= (in_x & in_y)
            
        return np.mean(inside_any) * w * h

    # Greedy Construction
    # Initialize positions with NaN
    best_x = np.full(num_tubes, np.nan)
    best_y = np.full(num_tubes, np.nan)
    
    # We will place tubes in the defined 'order'
    # To speed up, we will try a limited number of candidates for each position
    # and pick the one that minimizes the total volume.
    
    # Pre-compute candidate x,y pairs? 
    # There are 16*16 = 256 candidates. 
    # For each of 64 tubes, we try 256 candidates. 
    # Each evaluation is fast_volume on a growing set.
    # Total evaluations: 64 * 256 = 16384.
    # Each evaluation takes ~1ms? 16s total. This is feasible.
    
    # However, fast_volume is O(num_tubes * num_points).
    # As we add tubes, it gets slower.
    # Let's optimize: We can just check the new tube against the existing union?
    # No, the box (X0, X1, Y0, Y1) changes, which rescales the volume estimate.
    # So we must recalculate.
    
    # To make it faster, we can reduce num_search_pts during the greedy phase.
    num_search_pts_greedy = 5000
    
    def fast_volume_greedy(x, y, subset_idx=None):
        if subset_idx is not None:
            x_sub = x[subset_idx]
            y_sub = y[subset_idx]
            i_sub = i_flat[subset_idx]
            j_sub = j_flat[subset_idx]
        else:
            x_sub = x
            y_sub = y
            i_sub = i_flat
            j_sub = j_flat
            
        if len(x_sub) == 0:
            return 0.0

        X0 = np.min(x_sub)
        X1 = np.max(x_sub + (i_sub + 1) / N)
        Y0 = np.min(y_sub)
        Y1 = np.max(y_sub + (j_sub + 1) / N)
        
        w = X1 - X0
        h = Y1 - Y0
        if w <= 1e-9 or h <= 1e-9:
            return 0.0

        a = X0 + a_norm[:num_search_pts_greedy] * w
        b = Y0 + b_norm[:num_search_pts_greedy] * h
        
        inside_any = np.zeros(num_search_pts_greedy, dtype=bool)
        
        for k in range(len(x_sub)):
            bx = x_sub[k]
            by = y_sub[k]
            bi = i_sub[k]
            bj = j_sub[k]
            
            pc = c_samples[:num_search_pts_greedy]
            left = bx + pc * bi / N
            right = left + 1.0 / N
            bottom = by + pc * bj / N
            top = bottom + 1.0 / N
            
            in_x = (a >= left) & (a <= right)
            in_y = (b >= bottom) & (b <= top)
            
            inside_any |= (in_x & in_y)
            
        return np.mean(inside_any) * w * h

    # Greedy Loop
    placed_idx = []
    
    for step in range(num_tubes):
        if time.time() > deadline - 10:
            # Fallback: fill remaining with random
            remaining = [idx for idx in range(num_tubes) if idx not in placed_idx]
            for idx in remaining:
                best_x[idx] = np.random.rand()
                best_y[idx] = np.random.rand()
            break
            
        current_idx = order[step]
        placed_idx.append(current_idx)
        
        best_vol = np.inf
        best_cx = 0.0
        best_cy = 0.0
        
        # Try all grid candidates for this tube
        for cx in x_candidates:
            for cy in y_candidates:
                # Temporarily set position
                test_x = best_x.copy()
                test_y = best_y.copy()
                test_x[current_idx] = cx
                test_y[current_idx] = cy
                
                # Evaluate volume for all placed tubes so far
                vol = fast_volume_greedy(test_x, test_y, subset_idx=placed_idx)
                
                if vol < best_vol:
                    best_vol = vol
                    best_cx = cx
                    best_cy = cy
                    
        best_x[current_idx] = best_cx
        best_y[current_idx] = best_cy

    # After greedy construction, we have a complete set.
    # Now, do a local refinement to polish the result.
    # This combines the structural insight of greedy packing with local optimization.
    
    current_x = best_x.copy()
    current_y = best_y.copy()
    current_vol = fast_volume(current_x, current_y)
    
    best_x = current_x.copy()
    best_y = current_y.copy()
    best_vol = current_vol
    
    # Local Search Phase
    # Use a smaller step size and random restarts if stuck?
    # Just simple hill climbing with random restarts of step size.
    
    rng_local = np.random.default_rng(123)
    
    # Phase 1: Fine tuning
    step_sizes = [1.0/16, 1.0/32, 1.0/64]
    
    for ss in step_sizes:
        while time.time() < deadline - 5:
            improved = False
            # Try to move one tube
            k = rng_local.integers(0, num_tubes)
            axis = rng_local.integers(0, 2)
            
            delta = rng_local.choice([-1, 1]) * ss * rng_local.uniform(0.5, 1.5)
            
            new_x = current_x.copy()
            new_y = current_y.copy()
            
            if axis == 0:
                new_x[k] += delta
            else:
                new_y[k] += delta
            
            new_vol = fast_volume(new_x, new_y)
            
            if new_vol < current_vol:
                current_vol = new_vol
                current_x = new_x
                current_y = new_y
                improved = True
                
                if new_vol < best_vol:
                    best_vol = new_vol
                    best_x = new_x.copy()
                    best_y = new_y.copy()
            else:
                # No improvement, try another random move
                # To avoid getting stuck in a local minimum, we can occasionally accept worse moves
                # but for now, just continue.
                pass
                
            # If we don't improve in a while, we might want to break out of this step size
            # Simple counter:
            if not improved and rng_local.random() < 0.05:
                 # 5% chance to break if no improvement recently? 
                 # Let's just rely on the step size change.
                 pass

    # Final Check
    if not np.all(np.isfinite(best_x)) or not np.all(np.isfinite(best_y)):
        # Fallback to a simple grid
        best_x = i_flat / (2*N)
        best_y = j_flat / (2*N)
        
    return np.stack([best_x, best_y], axis=1)

# EVOLVE-BLOCK-END
