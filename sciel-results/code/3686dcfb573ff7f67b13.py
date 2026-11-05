import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Prepare boxes with original indices
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    # Sort by volume descending
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # bins: list of lists of (x, y, z, w, h, d)
    bins = []
    placements = [None] * n  # To store result in original order
    
    start_time = time.time()
    timeout = 100.0  # Safety margin
    
    for i, w, h, d, vol in indexed_boxes:
        if time.time() - start_time > timeout:
            # Fallback: simple shelf for remaining boxes
            # This is a safety net, shouldn't trigger often
            pass
            
        best_bin_idx = -1
        best_pos = None
        best_cost = float('inf')
        
        for b_idx, bin_contents in enumerate(bins):
            # Convert bin contents to numpy arrays for fast overlap check
            if len(bin_contents) == 0:
                # Empty bin, position (0,0,0) is always valid if it fits
                if w <= W and h <= H and d <= D:
                    cost = w * h * d  # BB volume
                    if cost < best_cost:
                        best_cost = cost
                        best_bin_idx = b_idx
                        best_pos = (0, 0, 0)
                continue
            
            # Extract coordinates
            # bin_contents is list of (x, y, z, w, h, d)
            placed = np.array(bin_contents, dtype=np.float64)
            px, py, pz, pw, ph, pd = placed[:,0], placed[:,1], placed[:,2], placed[:,3], placed[:,4], placed[:,5]
            
            # Generate candidate coordinates
            # x candidates: 0 and x_i + w_i
            cx = np.unique(np.concatenate(([0], px + pw)))
            cy = np.unique(np.concatenate(([0], py + ph)))
            cz = np.unique(np.concatenate(([0], pz + pd)))
            
            # Filter candidates that are within container
            cx = cx[cx + w <= W]
            cy = cy[cy + h <= H]
            cz = cz[cz + d <= D]
            
            if len(cx) == 0 or len(cy) == 0 or len(cz) == 0:
                continue
                
            # We want to find the position that minimizes the bounding box volume of the bin.
            # BB volume = (max_x + w) * (max_y + h) * (max_z + d) is not quite right.
            # The BB of the bin is determined by the max extent of ALL boxes.
            # If we place a new box, the new BB is:
            # max_x_new = max(max(px+pw), x+w)
            # max_y_new = max(max(py+ph), y+h)
            # max_z_new = max(max(pz+pd), z+d)
            # Cost = max_x_new * max_y_new * max_z_new
            
            current_max_x = np.max(px + pw)
            current_max_y = np.max(py + ph)
            current_max_z = np.max(pz + pd)
            
            best_local_pos = None
            best_local_cost = float('inf')
            
            # Iterate candidates
            # To speed up, we can limit the number of candidates or use a heuristic.
            # With small N, full iteration is okay.
            for x in cx:
                if x + w > W: continue
                for y in cy:
                    if y + h > H: continue
                    for z in cz:
                        if z + d > D: continue
                        
                        # Check overlap
                        # Overlap if:
                        # x < px_i + pw_i AND px_i < x + w
                        # y < py_i + ph_i AND py_i < y + h
                        # z < pz_i + pd_i AND pz_i < z + d
                        
                        overlap_x = (px < x + w) & (px + pw > x)
                        overlap_y = (py < y + h) & (py + ph > y)
                        overlap_z = (pz < z + d) & (pz + pd > z)
                        
                        if np.any(overlap_x & overlap_y & overlap_z):
                            continue
                        
                        # Calculate cost
                        new_max_x = max(current_max_x, x + w)
                        new_max_y = max(current_max_y, y + h)
                        new_max_z = max(current_max_z, z + d)
                        cost = new_max_x * new_max_y * new_max_z
                        
                        if cost < best_local_cost:
                            best_local_cost = cost
                            best_local_pos = (x, y, z)
                            
            if best_local_pos is not None:
                if best_local_cost < best_cost:
                    best_cost = best_local_cost
                    best_bin_idx = b_idx
                    best_pos = best_local_pos
                    
        if best_bin_idx == -1:
            # Open new bin
            bins.append([])
            best_bin_idx = len(bins) - 1
            best_pos = (0, 0, 0)
            
        # Place box
        x, y, z = best_pos
        bins[best_bin_idx].append((x, y, z, w, h, d))
        placements[i] = (best_bin_idx, x, y, z, w, h, d)
        
    return placements
