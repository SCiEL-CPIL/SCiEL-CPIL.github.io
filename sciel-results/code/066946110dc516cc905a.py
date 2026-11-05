import numpy as np
from scipy import ndimage

def pack(boxes, container):
    W, H, D = container
    
    # Sort boxes by volume descending to place larger items first
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    bins = []  # List of (grid, list_of_placements)
    placements_by_index = [None] * len(boxes)
    
    def try_place_in_bin(bin_idx, w, h, d):
        """Try to place a box in a specific bin. Returns (x,y,z) or None."""
        grid = bins[bin_idx][0]
        
        # Find all maximal empty rectangles using ndimage
        # We need to find positions where a box of size (w,h,d) can fit.
        # This is equivalent to finding a sub-array of size (w,h,d) that is all False.
        
        # Efficient approach: For each possible z-layer, check if there's a 2D empty rectangle
        # that can support the depth d.
        
        # Alternative: Use convolution or simple sliding window for small grids.
        # Given the constraint of 120s and 40 boxes, grids are likely small enough for
        # a optimized search or a simple scan.
        
        # Let's use a simple scan but optimized with numpy slicing.
        # We look for the first (x,y,z) where grid[x:x+w, y:y+h, z:z+d] is all False.
        
        # To make this faster, we can precompute "free space" projections or just scan.
        # Since W,H,D are not specified but likely moderate, let's try a smart scan.
        
        # We'll iterate z from 0 to D-d, y from 0 to H-h, x from 0 to W-w
        # But we can do better by checking 2D slices.
        
        for z in range(D - d + 1):
            # Check if the slice z:z+d is completely empty in some 2D region
            # Actually, we need a 3D block.
            # Let's check if there is any x,y such that grid[x:x+w, y:y+h, z:z+d] is all False.
            
            # We can compute a 2D mask of where the 3D block would be empty.
            # This is equivalent to: for all dx in [0,w), dy in [0,h), dz in [0,d), 
            # grid[x+dx, y+dy, z+dz] == False
            
            # This can be done by taking the min over the 3D block.
            # But doing this for every x,y,z is expensive.
            
            # Instead, let's just do a linear scan for simplicity and correctness first,
            # as the grid size is likely small (e.g., 10x10x10).
            # If performance is an issue, we can optimize later.
            
            for y in range(H - h + 1):
                for x in range(W - w + 1):
                    if not grid[x:x+w, y:y+h, z:z+d].any():
                        return (x, y, z)
        return None

    def get_bin_score(bin_idx, w, h, d, pos):
        """Calculate a score for a placement. Lower is better."""
        if pos is None:
            return float('inf')
        x, y, z = pos
        # Prefer lower z, then lower y, then lower x (shelf-like)
        return z * 10000 + y * 100 + x

    for orig_idx, (w, h, d) in indexed_boxes:
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        # Try to place in existing bins
        for b_idx in range(len(bins)):
            pos = try_place_in_bin(b_idx, w, h, d)
            if pos is not None:
                score = get_bin_score(b_idx, w, h, d, pos)
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        # If not found in existing bins, try to create a new bin
        if best_bin == -1:
            # Check if it fits in a new bin
            if w <= W and h <= H and d <= D:
                best_bin = len(bins)
                best_pos = (0, 0, 0)
                # Initialize new bin
                grid = np.zeros((W, H, D), dtype=bool)
                bins.append((grid, []))
        
        # If still not placed, it's an error (shouldn't happen if boxes fit)
        if best_bin == -1:
            # Fallback: place in bin 0 at 0,0,0 (will cause overlap, but avoids crash)
            # Actually, the problem states boxes fit. Let's assume they do.
            # If a box is too large, we can't place it. But the task implies valid instances.
            # For safety, we'll just append it to the first bin at 0,0,0.
            best_bin = 0 if bins else 0
            best_pos = (0, 0, 0)
            if not bins:
                grid = np.zeros((W, H, D), dtype=bool)
                bins.append((grid, []))
        
        # Place the box
        grid, bin_placements = bins[best_bin]
        x, y, z = best_pos
        grid[x:x+w, y:y+h, z:z+d] = True
        bin_placements.append((orig_idx, x, y, z, w, h, d))
        placements_by_index[orig_idx] = (best_bin, x, y, z, w, h, d)
    
    # Construct final list of placements in original order
    final_placements = []
    for i in range(len(boxes)):
        final_placements.append(placements_by_index[i])
        
    return final_placements
