"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    """
    3D bin packing using Best-Fit Decreasing with shelf-based placement.
    Returns list of (bin_id, x, y, z, w, h, d) for each box.
    """
    deadline = time.time() + 110.0  # 110s safety margin
    
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    indices = list(range(n))
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    indices.sort(key=lambda i: -volumes[i])
    
    # Bins: each bin has a list of placed boxes (x, y, z, w, h, d)
    bins = []  # list of lists
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) without overlap and within container."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Check overlap
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True
    
    def find_positions(bin_boxes, w, h, d):
        """Find all valid (x, y, z) positions for a box in a bin.
        Returns list of positions sorted by a heuristic (lower is better)."""
        positions = []
        
        # Generate candidate positions based on existing boxes and container boundaries
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            if bx + bw <= W:
                x_candidates.add(bx + bw)
            if by + bh <= H:
                y_candidates.add(by + bh)
            if bz + bd <= D:
                z_candidates.add(bz + bd)
        
        # Also add container boundaries minus box dimensions
        if w <= W:
            x_candidates.add(W - w)
        if h <= H:
            y_candidates.add(H - h)
        if d <= D:
            z_candidates.add(D - d)
        
        # Filter candidates that are within bounds
        x_candidates = [x for x in x_candidates if x >= 0 and x + w <= W]
        y_candidates = [y for y in y_candidates if y >= 0 and y + h <= H]
        z_candidates = [z for z in z_candidates if z >= 0 and z + d <= D]
        
        # Try all combinations
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        # Score: prefer lower x, then lower y, then lower z
                        score = x * 10000 + y * 100 + z
                        positions.append((score, x, y, z))
        
        positions.sort()
        return [(x, y, z) for (_, x, y, z) in positions]
    
    # Track which box index goes to which bin
    placement_map = [None] * n
    
    for idx in indices:
        w, h, d = boxes[idx]
        placed = False
        
        # Try existing bins first (best fit)
        best_bin = -1
        best_pos = None
        best_score = None
        
        for bi in range(len(bins)):
            positions = find_positions(bins[bi], w, h, d)
            if positions:
                # Pick the first position (lowest score)
                x, y, z = positions[0]
                # Score for bin selection: prefer bins where the box fits with minimal "waste"
                # Use the position score as a proxy
                pos_score = x * 10000 + y * 100 + z
                if best_score is None or pos_score < best_score:
                    best_score = pos_score
                    best_bin = bi
                    best_pos = (x, y, z)
        
        if not placed and best_bin >= 0:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            placement_map[idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
        else:
            # Open a new bin
            new_bin = len(bins)
            bins.append([])
            # Place at origin
            bins[new_bin].append((0, 0, 0, w, h, d))
            placement_map[idx] = (new_bin, 0, 0, 0, w, h, d)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Handle any unplaced boxes (shouldn't happen, but just in case)
    for idx in range(n):
        if placement_map[idx] is None:
            # Force place in a new bin
            new_bin = len(bins)
            bins.append([])
            w, h, d = boxes[idx]
            bins[new_bin].append((0, 0, 0, w, h, d))
            placement_map[idx] = (new_bin, 0, 0, 0, w, h, d)
    
    # Build result in original order
    placements = []
    for idx in range(n):
        b, x, y, z, w, h, d = placement_map[idx]
        placements.append((b, x, y, z, w, h, d))
    
    return placements
# EVOLVE-BLOCK-END
