"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, but keep track of original indices
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # We will maintain a list of bins. Each bin has a list of placed boxes (w, h, d, x, y, z).
    # For efficient packing, we use a "skyline" or "shelf" approach per bin, but since we can't rotate,
    # a simple shelf heuristic per bin with First Fit Decreasing is a good start.
    # To improve over the single shelf, we can try multiple shelves and pick the best.
    
    # Actually, a more robust approach for 3D packing without rotation is to use a 
    # "Best Fit" or "First Fit" with a simple collision check, but that's O(n^2 * bins).
    # Given the small size (40 boxes), we can afford a slightly more complex heuristic.
    
    # Let's use a First Fit Decreasing with a simple "place at lowest possible z, then y, then x" 
    # strategy for each bin. This is essentially a "corner" placement.
    
    def can_place(bin_boxes, w, h, d, x, y, z):
        for (bw, bh, bd, bx, by, bz) in bin_boxes:
            # Check overlap
            if (x < bx + bw and x + w > bx and
                y < by + bh and y + h > by and
                z < bz + bd and z + d > bz):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        # Try to find a position (x, y, z) in [0, W-w] x [0, H-h] x [0, D-d]
        # We prioritize lower z, then lower y, then lower x.
        # To make it faster, we can iterate over possible z, y, x from 0 upwards.
        # But this is O(W*H*D) which is too slow if dimensions are large.
        # Instead, we can use a "skyline" or "max height" approach.
        # For each (x, y), we maintain the maximum z height occupied.
        # However, since boxes have different depths, this is tricky.
        
        # A simpler approach for small n: iterate over existing boxes and try to place
        # adjacent to them, or at the origin.
        # Given the constraints, let's use a simple greedy:
        # 1. Try (0, 0, 0)
        # 2. For each existing box, try placing the new box on top (z = bz + bd), 
        #    to the right (x = bx + bw), or in front (y = by + bh), if within bounds and no overlap.
        
        candidates = [(0, 0, 0)]
        for (bw, bh, bd, bx, by, bz) in bin_boxes:
            # On top
            if bz + bd + d <= D:
                candidates.append((bx, by, bz + bd))
            # To the right
            if bx + bw + w <= W:
                candidates.append((bx + bw, by, bz))
            # In front
            if by + bh + h <= H:
                candidates.append((bx, by + bh, bz))
        
        # Also try some other common positions
        # For now, just sort candidates by z, y, x and find the first valid one.
        candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for (x, y, z) in candidates:
            if x < 0 or y < 0 or z < 0:
                continue
            if x + w > W or y + h > H or z + d > D:
                continue
            if can_place(bin_boxes, w, h, d, x, y, z):
                return (x, y, z)
        return None
    
    bins = []  # List of lists of (w, h, d, x, y, z)
    
    for i, w, h, d in indexed_boxes:
        placed = False
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                bin_boxes.append((w, h, d, x, y, z))
                placed = True
                break
        if not placed:
            # Start a new bin
            bins.append([(w, h, d, 0, 0, 0)])
    
    # Build the result in the original order
    # We need to map back to original indices.
    # The `indexed_boxes` list has (orig_idx, w, h, d).
    # We need to record which bin and position each original box got.
    
    # Let's redo this properly, tracking original indices.
    bins = []  # List of dicts: {orig_idx: (w, h, d, x, y, z)}
    
    for orig_idx, w, h, d in indexed_boxes:
        placed = False
        for b_idx in range(len(bins)):
            bin_boxes = list(bins[b_idx].values())
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx][orig_idx] = (w, h, d, x, y, z)
                placed = True
                break
        if not placed:
            new_bin = {orig_idx: (w, h, d, 0, 0, 0)}
            bins.append(new_bin)
    
    placements = [None] * n
    for b_idx, bin_dict in enumerate(bins):
        for orig_idx, (w, h, d, x, y, z) in bin_dict.items():
            placements[orig_idx] = (b_idx, x, y, z, w, h, d)
    
    return placements

# EVOLVE-BLOCK-END
