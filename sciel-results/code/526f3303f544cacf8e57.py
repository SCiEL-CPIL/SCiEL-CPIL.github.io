"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy import ndimage

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending to place large items first (better for bin count)
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    bins = []  # list of occupancy grids (numpy boolean arrays)
    placements = [None] * len(boxes)
    
    def try_place(box, bin_idx):
        """Try to place box in bin_idx using maximal empty space heuristic.
        Returns (x, y, z) if successful, None otherwise."""
        w, h, d = box
        grid = bins[bin_idx]
        
        # We need to find a position (x, y, z) such that grid[x:x+w, y:y+h, z:z+d] is all False.
        # Using a sliding window approach with ndimage might be complex for arbitrary box sizes.
        # Instead, we'll use a heuristic that scans for the "best" empty spot.
        # For efficiency, we can pre-compute a "free space" map or use a simpler heuristic.
        
        # Heuristic: Find the position that minimizes the "corner" distance (x, y, z)
        # to keep the packing compact. We scan x, then y, then z.
        # To speed up, we can check if a box fits by using a summed area table or similar,
        # but for 40 boxes and moderate container sizes, a direct check with early termination
        # in a smart order is sufficient.
        
        # Let's define a cost function: minimize x, then y, then z.
        # We iterate through possible positions in a specific order.
        
        # Optimization: Use a 3D array of "max extent" to quickly rule out areas?
        # Simpler: Just scan. The container dimensions are likely not huge.
        
        best_pos = None
        best_cost = None
        
        # Scan x from 0 to W-w
        for x in range(W - w + 1):
            for y in range(H - h + 1):
                for z in range(D - d + 1):
                    # Check if the sub-cube is free
                    # grid[x:x+w, y:y+h, z:z+d].any() should be False
                    if not np.any(grid[x:x+w, y:y+h, z:z+d]):
                        # Found a valid position
                        # Cost: prioritize lower x, then lower y, then lower z
                        # This keeps items towards the origin, filling space compactly.
                        cost = (x, y, z)
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y, z)
                        # Since we scan x, y, z in increasing order, the first one found
                        # is the minimum in lexicographical order. We can break early.
                        return best_pos
        
        return None

    def find_best_bin(box):
        """Find the best existing bin to place the box in.
        Returns (bin_idx, x, y, z) or None."""
        best_bin_idx = None
        best_pos = None
        best_cost = None
        
        for i in range(len(bins)):
            pos = try_place(box, i)
            if pos is not None:
                x, y, z = pos
                # Cost for choosing this bin:
                # 1. Prefer bins that are already open (all bins in 'bins' are open)
                # 2. Among open bins, prefer the one that minimizes wasted space?
                #    Or simply the first one that fits?
                #    To minimize B, we should reuse bins.
                #    To minimize last-bin fraction, we should pack tightly.
                #    Let's use a cost that favors bins with more existing volume?
                #    Actually, the simplest is to just pick the first bin that fits,
                #    but we want to be smart.
                #    Let's score by the position cost (x,y,z) to keep it compact.
                #    But also, we should prefer bins that are "fuller" to avoid opening new ones?
                #    No, all bins in 'bins' are already open. So any bin is "free" in terms of B.
                #    The goal is to minimize the last bin's usage.
                #    So we should pack the last bin as full as possible?
                #    Or pack all bins as full as possible?
                #    Since B is the number of bins, we must reuse existing bins.
                #    The tie-breaker is the last-bin fraction.
                #    So, we should prefer to place items in the last bin if it doesn't make it worse?
                #    Actually, if we place in an earlier bin, the last bin remains the same.
                #    If we place in the last bin, the last bin fraction increases (good).
                #    So we should prefer the last bin?
                #    Wait, if we place in an earlier bin, we might free up space in the last bin for future items?
                #    No, items are placed in order.
                #    Let's just use the position cost (x,y,z) to keep packing compact.
                #    This generally leads to better bin utilization.
                
                cost = (x, y, z)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_bin_idx = i
                    best_pos = pos
        
        if best_bin_idx is not None:
            return best_bin_idx, best_pos
        return None

    for orig_idx, box in indexed_boxes:
        w, h, d = box
        
        # Try to place in existing bins
        result = find_best_bin(box)
        
        if result is not None:
            bin_idx, (x, y, z) = result
            # Update grid
            bins[bin_idx][x:x+w, y:y+h, z:z+d] = True
            placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
        else:
            # Open a new bin
            new_bin_idx = len(bins)
            new_grid = np.zeros((W, H, D), dtype=bool)
            bins.append(new_grid)
            
            # Place at (0,0,0) in the new bin
            x, y, z = 0, 0, 0
            new_grid[x:x+w, y:y+h, z:z+d] = True
            placements[orig_idx] = (new_bin_idx, x, y, z, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
