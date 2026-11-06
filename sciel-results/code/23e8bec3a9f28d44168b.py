"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n_boxes = len(boxes)
    
    # Sort boxes by volume descending
    box_volumes = [w*h*d for w,h,d in boxes]
    sorted_indices = np.argsort(box_volumes)[::-1]
    
    placements = [None] * n_boxes
    bins = [] # List of dicts: {'grid': np.array, 'vol': int}
    
    for orig_idx in sorted_indices:
        w, h, d = boxes[orig_idx]
        placed = False
        
        for bin_idx, bin_data in enumerate(bins):
            # Check if box fits in bin volume-wise
            if bin_data['vol'] + w*h*d > W*H*D:
                continue
            
            # Find best position using efficient 3D sliding window check
            grid = bin_data['grid']
            found_pos = None
            
            # We want to find (x, y, z) such that grid[x:x+w, y:y+h, z:z+d] is all 0.
            # We iterate z from 0 to D-d.
            # For each z, we compute a 2D array of "occupied sums" in the z-slice.
            # Then we find if any (x,y) window of size (w,h) has sum 0.
            
            # Optimization: Pre-compute cumulative sums for the grid?
            # Or just use sliding_window_view on the z-slice sum.
            
            for z in range(D - d + 1):
                # Sum over the z-dimension for the window height d
                # slice shape: (W, H, d)
                # sum_z shape: (W, H)
                # If sum_z[x, y] > 0, then voxel (x, y, z..z+d-1) has some occupancy.
                # We need a (w, h) window in sum_z to be all 0.
                
                # Fast check: if the total sum in the entire z-slice is 0, then any (x,y) works.
                # But we want the "best" position (lowest x, then y).
                
                # Compute sum_z efficiently
                # grid[:, :, z:z+d] is a view
                # np.sum(..., axis=2)
                
                # To speed up, we can check if the box fits in the remaining volume of this z-layer?
                # No, that's not sufficient due to spatial distribution.
                
                # Let's use the sliding window view on the 2D sum.
                if W >= w and H >= h:
                    # sum_z = np.sum(grid[:, :, z:z+d], axis=2)
                    # This creates a new array of shape (W, H).
                    
                    # Alternative: Use integral image (prefix sum) of the 3D grid?
                    # That would be expensive to update.
                    
                    # Let's stick to the sliding window view on the 2D projection.
                    sum_z = np.sum(grid[:, :, z:z+d], axis=2)
                    
                    # Check if there is any (x, y) where the (w, h) window sum is 0.
                    # sliding_window_view(sum_z, (w, h)) gives shape (W-w+1, H-h+1, w, h)
                    # Sum over last two axes.
                    
                    # Optimization: If the global sum of sum_z is 0, then (0,0) is valid.
                    # If not, we need to search.
                    
                    # To avoid creating the full sliding window view if possible:
                    # We can check if min(sum_z) is 0? No, that's not enough.
                    
                    # Let's try to find the first valid position.
                    # We can iterate x from 0 to W-w:
                    #   Check if there is any y such that sum_z[x:x+w, y:y+h] sum is 0.
                    #   This is still a 1D sliding window.
                    
                    # Given the small size, let's just use the vectorized approach.
                    try:
                        # Create sliding window view
                        windows = np.lib.stride_tricks.sliding_window_view(sum_z, (w, h))
                        # windows shape: (W-w+1, H-h+1, w, h)
                        # Sum over the window dimensions
                        window_sums = np.sum(windows, axis=(2, 3))
                        
                        # Find the first position where window_sum is 0
                        # We want smallest z, then smallest x, then smallest y.
                        # np.argwhere returns coordinates in row-major order (x then y).
                        valid = window_sums == 0
                        if np.any(valid):
                            # Find the first True
                            coords = np.argwhere(valid)
                            # coords[0] is the first one (smallest x, then smallest y)
                            x, y = coords[0]
                            found_pos = (x, y, z)
                            break
                    except Exception:
                        pass
            
            if found_pos:
                x, y, z = found_pos
                # Place box
                grid[x:x+w, y:y+h, z:z+d] = 1
                bin_data['vol'] += w*h*d
                placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Create new bin
            grid = np.zeros((W, H, D), dtype=np.uint8)
            grid[0:w, 0:h, 0:d] = 1
            bins.append({'grid': grid, 'vol': w*h*d})
            placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
    
    return placements

# EVOLVE-BLOCK-END
