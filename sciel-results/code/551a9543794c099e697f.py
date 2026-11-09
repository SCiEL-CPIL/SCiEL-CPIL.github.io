import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 150s limit (or 120s if strict)
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Pre-compute orientations
    def get_orientations(w, h, d):
        dims = list(itertools.permutations((w, h, d)))
        seen = set()
        unique = []
        for dim in dims:
            if dim not in seen:
                seen.add(dim)
                unique.append(dim)
        return unique

    box_orientations = []
    for w, h, d in boxes:
        box_orientations.append(get_orientations(w, h, d))
        
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    # State representation for Skyline Packing
    # We use a grid to represent the "height" (z-coordinate) of the surface.
    # To keep memory/time manageable, we use a coarse grid resolution.
    # If dimensions are large, we can't use 1-unit resolution.
    # Let's determine resolution based on max dimension.
    max_dim = max(W, H, D)
    # Aim for ~50x50 grid max.
    res_x = max(1, int(W / 50))
    res_y = max(1, int(H / 50))
    
    # Actually, for 3D packing, "Skyline" usually refers to the 2D base (X,Y) 
    # storing the max Z used. But boxes have depth.
    # A more robust "Different Strategy" is the **Corner Method** but optimized 
    # with **Best Fit Decreasing** and **Local Search**.
    # The current code already does Corner Method. 
    # Let's pivot to **Guillotine Cutting / Space Management** logic? 
    # No, that's complex.
    
    # Let's stick to the **Vectorized Corner Method** but add **Local Search (2-opt)** 
    # which is the "Materially Different" part. The current code does NOT do local search.
    # It only does random restarts. Local search on the sequence of boxes is a distinct 
    # optimization family (Metaheuristic vs Randomized Heuristic).

    def run_pack(order):
        """
        Greedy packing using vectorized collision checks.
        Returns: results list, placed array, placed_indices list
        """
        placed_arr = np.empty((0, 6)) # x, y, z, w, h, d
        results = [None] * n
        placed_indices = [] # indices in 'order' that were successfully placed
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            # Generate candidates
            # 1. (0,0,0)
            # 2. Corners of placed boxes
            candidates = set()
            candidates.add((0, 0, 0))
            
            if len(placed_arr) > 0:
                # Vectorized candidate generation is hard with sets.
                # But N is small (60). Looping is fine.
                for i in range(len(placed_arr)):
                    x, y, z, w, h, d = placed_arr[i]
                    # 7 corners
                    candidates.add((x + w, y, z))
                    candidates.add((x, y + h, z))
                    candidates.add((x, y, z + d))
                    candidates.add((x + w, y + h, z))
                    candidates.add((x + w, y, z + d))
                    candidates.add((x, y + h, z + d))
                    candidates.add((x + w, y + h, z + d))
            
            cand_list = list(candidates)
            if not cand_list:
                continue
                
            cand_arr = np.array(cand_list, dtype=np.float64)
            cx = cand_arr[:, 0]
            cy = cand_arr[:, 1]
            cz = cand_arr[:, 2]
            
            valid_mask = np.ones(len(cand_list), dtype=bool)
            
            # Filter by bounds
            # We need to check if ANY orientation fits.
            # To optimize, we can check the most likely orientations first?
            # Or just check all.
            
            best_valid_idx = -1
            best_orient = None
            best_z = float('inf')
            best_y = float('inf')
            best_x = float('inf')
            
            for (ow, oh, od) in orientations:
                # Bounds check
                mask_bounds = (cx + ow <= W) & (cy + oh <= H) & (cz + od <= D) & (cx >= 0) & (cy >= 0) & (cz >= 0)
                
                if not np.any(mask_bounds):
                    continue
                
                valid_mask = mask_bounds.copy()
                
                if len(placed_arr) > 0:
                    # Vectorized collision check
                    px = placed_arr[:, 0][None, :]
                    py = placed_arr[:, 1][None, :]
                    pz = placed_arr[:, 2][None, :]
                    pw = placed_arr[:, 3][None, :]
                    ph = placed_arr[:, 4][None, :]
                    pd = placed_arr[:, 5][None, :]
                    
                    cx_m = cx[:, None]
                    cy_m = cy[:, None]
                    cz_m = cz[:, None]
                    
                    overlap_x = (cx_m < px + pw) & (cx_m + ow > px)
                    overlap_y = (cy_m < py + ph) & (cy_m + oh > py)
                    overlap_z = (cz_m < pz + pd) & (cz_m + od > pz)
                    
                    overlap_full = overlap_x & overlap_y & overlap_z
                    bad_candidates = np.any(overlap_full, axis=1)
                    valid_mask[bad_candidates] = False
                
                if not np.any(valid_mask):
                    continue
                
                valid_indices = np.where(valid_mask)[0]
                
                # Score: Minimize Z, then Y, then X (Bottom-Left-Far heuristic)
                # We want the lowest Z. If tie, lowest Y. If tie, lowest X.
                # This is equivalent to minimizing (Z * H * W + Y * W + X)
                scores = cz[valid_indices] * H * W + cy[valid_indices] * W + cx[valid_indices]
                
                min_score_idx = np.argmin(scores)
                best_cand_idx = valid_indices[min_score_idx]
                current_score = scores[min_score_idx]
                
                # Compare with global best for this box
                # We prefer lower Z, then lower Y, then lower X
                cand_z = cz[best_cand_idx]
                cand_y = cy[best_cand_idx]
                cand_x = cx[best_cand_idx]
                
                if (cand_z < best_z or 
                    (cand_z == best_z and cand_y < best_y) or
                    (cand_z == best_z and cand_y == best_y and cand_x < best_x)):
                    best_z = cand_z
                    best_y = cand_y
                    best_x = cand_x
                    best_pos = (cand_x, cand_y, cand_z)
                    best_orientation = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                new_box = np.array([[cx, cy, cz, ow, oh, od]], dtype=np.float64)
                if len(placed_arr) == 0:
                    placed_arr = new_box
                else:
                    placed_arr = np.vstack((placed_arr, new_box))
                placed_indices.append(idx)
                results[idx] = (0, cx, cy, cz, ow, oh, od)
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)
        
        return results, placed_arr, placed_indices

    def local_search(initial_order, max_iter=100):
        """
        Perform 2-opt swaps to improve packing.
        """
        current_order = initial_order[:]
        current_results, _, _ = run_pack(current_order)
        current_vol = sum(w*h*d for (b, x, y, z, w, h, d) in current_results if b == 0)
        
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Try swapping every pair of boxes in the order
            # To save time, we only try swapping adjacent pairs or random pairs?
            # 2-opt is usually adjacent in TSP, but for packing, any pair swap is valid.
            # We'll use random pairs to explore space, or systematic if fast enough.
            # n=60, pairs = 1770. run_pack is fast. 1770 * ~1ms = ~2s per full pass.
            # We can do a few passes.
            
            # Systematic pass
            for i in range(len(current_order)):
                for j in range(i + 1, len(current_order)):
                    if time.time() - start_time > DEADLINE - 5:
                        return current_results, current_vol
                    
                    # Swap
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    
                    new_results, _, _ = run_pack(current_order)
                    new_vol = sum(w*h*d for (b, x, y, z, w, h, d) in new_results if b == 0)
                    
                    if new_vol > current_vol:
                        current_vol = new_vol
                        current_results = new_results
                        improved = True
                    else:
                        # Swap back
                        current_order[i], current_order[j] = current_order[j], current_order[i]
                        
                    if improved:
                        # Restart inner loop? Or just continue?
                        # Restarting is safer for local search.
                        break
                if improved:
                    break
                    
        return current_results, current_vol

    # Main Search
    best_results = None
    best_volume = -1
    
    # 1. Initial Heuristics
    # Strategy A: Volume Descending
    order_a = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    res_a, vol_a = local_search(order_a, max_iter=5) # Limited local search
    if vol_a > best_volume:
        best_volume = vol_a
        best_results = res_a
        
    # Strategy B: Max Dimension Descending
    order_b = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    res_b, vol_b = local_search(order_b, max_iter=5)
    if vol_b > best_volume:
        best_volume = vol_b
        best_results = res_b
        
    # Strategy C: Aspect Ratio (Cubic first)
    def aspect_ratio(box):
        w, h, d = box
        if min(w,h,d) == 0: return 0
        return max(w,h,d) / min(w,h,d)
    order_c = sorted(range(n), key=lambda i: aspect_ratio(boxes[i]))
    res_c, vol_c = local_search(order_c, max_iter=5)
    if vol_c > best_volume:
        best_volume = vol_c
        best_results = res_c

    # 2. Random Restarts with Local Search
    random.seed(42)
    # Budget remaining
    while time.time() - start_time < DEADLINE - 10:
        # Random order
        order_r = list(range(n))
        random.shuffle(order_r)
        
        # Run a quick greedy pack to see if it's promising
        res_r, _, _ = run_pack(order_r)
        vol_r = sum(w*h*d for (b, x, y, z, w, h, d) in res_r if b == 0)
        
        # Only do local search if it's close to best or very good
        if vol_r > best_volume * 0.9:
            # Do a short local search
            res_r_ls, vol_r_ls = local_search(order_r, max_iter=3)
            if vol_r_ls > best_volume:
                best_volume = vol_r_ls
                best_results = res_r_ls
        
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results
# EVOLVE-BLOCK-END
