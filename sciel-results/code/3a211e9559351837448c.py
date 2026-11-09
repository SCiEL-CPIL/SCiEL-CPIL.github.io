"""Improved 3D bin packing using Multi-Start Next-Fit Shelf Packing."""
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    def run_shelf_packing(sorted_indices):
        bins = [] # List of bins. Each bin is a list of shelves.
        # Shelf: [z_start, current_max_height, x_used]
        placements = [None] * n
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            placed = False
            
            for bin_idx in range(len(bins)):
                shelves = bins[bin_idx]
                
                # Try to place in the last shelf of this bin
                if shelves:
                    last_shelf = shelves[-1]
                    z_start, shelf_h, x_used = last_shelf
                    
                    # Check if it fits in x
                    if x_used + w <= W:
                        # Check if it fits in z (vertical space in bin)
                        # The box sits at z_start. Its top is z_start + h.
                        # Must be <= D.
                        if z_start + h <= D:
                            # Place it
                            x_pos = x_used
                            y_pos = 0
                            z_pos = z_start
                            
                            # Update shelf
                            last_shelf[2] = x_used + w
                            if h > shelf_h:
                                last_shelf[1] = h
                                
                            placements[idx] = (bin_idx, x_pos, y_pos, z_pos, w, h, d)
                            placed = True
                            break
                            
                # If it didn't fit in the last shelf (or bin was empty), try to start a new shelf
                if not placed:
                    if shelves:
                        last_shelf = shelves[-1]
                        new_z_start = last_shelf[0] + last_shelf[1]
                    else:
                        new_z_start = 0
                        
                    if new_z_start + h <= D:
                        # Create new shelf
                        shelves.append([new_z_start, h, w])
                        placements[idx] = (bin_idx, 0, 0, new_z_start, w, h, d)
                        placed = True
                        break
                        
            if not placed:
                # Start new bin
                bins.append([[0, h, w]])
                placements[idx] = (len(bins)-1, 0, 0, 0, w, h, d)
                
        return placements, len(bins)

    # Define sorting strategies
    # 1. Volume descending
    # 2. Height descending
    # 3. Max dimension descending
    # 4. Volume descending, but tie-break by height descending
    
    strategies = [
        lambda i: (-boxes[i][0]*boxes[i][1]*boxes[i][2]),
        lambda i: (-boxes[i][1]),
        lambda i: (-max(boxes[i])),
        lambda i: (-boxes[i][0]*boxes[i][1]*boxes[i][2], -boxes[i][1])
    ]
    
    best_placements = None
    best_score = float('inf')
    
    for strategy in strategies:
        sorted_indices = sorted(range(n), key=strategy)
        placements, num_bins = run_shelf_packing(sorted_indices)
        
        if not placements:
            continue
            
        last_bin_idx = max(p[0] for p in placements)
        
        last_bin_vol = 0
        for p in placements:
            if p[0] == last_bin_idx:
                last_bin_vol += p[4] * p[5] * p[6]
                
        container_vol = W * H * D
        if container_vol == 0:
            fill_ratio = 1.0
        else:
            fill_ratio = last_bin_vol / container_vol
            
        score = num_bins + 0.1 * (1 - fill_ratio)
        
        if score < best_score:
            best_score = score
            best_placements = placements
            
    if best_placements is None:
        return [(0, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]
        
    return best_placements
