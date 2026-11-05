"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending, then by max dimension descending
    # We need to preserve original dimensions (no rotation), so w,h,d are fixed
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -max(x[1])))
    
    # We'll maintain a list of bins, each with a 3D boolean occupancy grid
    # But for performance, we might want to use a more efficient representation
    # Let's use a list of (bin_id, grid) where grid is a 3D numpy array
    
    bins = []  # list of 3D boolean arrays
    placements = [None] * len(boxes)
    
    def try_place_in_bin(bin_grid, w, h, d):
        """
        Try to find a position in the bin for a box of size (w, h, d).
        Returns (x, y, z) if placed, None otherwise.
        Uses a 'first fit' strategy scanning from origin, with a preference for
        positions that minimize wasted space (closest to origin).
        """
        # Scan all possible positions
        # To speed up, we can check if there's any valid position by scanning
        # We'll scan in order of increasing x, y, z (closest to origin first)
        # This tends to fill corners well
        
        # For efficiency, we can use numpy to check all positions at once
        # but that might be memory intensive for large containers.
        # Given 40 boxes, let's try a hybrid: scan x, y, z but use numpy for the overlap check
        
        # Actually, let's just do a simple scan since W, H, D are likely not huge
        # and we only have 40 boxes.
        
        for x in range(W - w + 1):
            for y in range(H - h + 1):
                for z in range(D - d + 1):
                    # Check if the box fits
                    if not np.any(bin_grid[x:x+w, y:y+h, z:z+d]):
                        bin_grid[x:x+w, y:y+h, z:z+d] = True
                        return (x, y, z)
        return None
    
    # Alternative: use a more efficient approach with "maximal empty box" search
    # But for now, let's try the simple first-fit with origin preference
    
    for orig_idx, (w, h, d) in indexed_boxes:
        placed = False
        
        # Try to place in existing bins
        # Sort bins by how full they are (most full first) to encourage consolidation
        # Actually, let's try bins in order of creation first, then by fill level
        
        # Try each existing bin
        for bin_idx in range(len(bins)):
            pos = try_place_in_bin(bins[bin_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
                placed = True
                break
        
        # If not placed, create a new bin
        if not placed:
            bin_idx = len(bins)
            grid = np.zeros((W, H, D), dtype=bool)
            bins.append(grid)
            pos = try_place_in_bin(grid, w, h, d)
            if pos is not None:
                x, y, z = pos
                placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
            else:
                # This shouldn't happen if the box fits in an empty container
                # But just in case, raise an error or handle gracefully
                raise ValueError(f"Box {orig_idx} with dimensions ({w},{h},{d}) does not fit in container ({W},{H},{D})")
    
    return placements

# EVOLVE-BLOCK-END
