"""Improved 3D bin packing using Best-Fit Decreasing with efficient bin management."""
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume in descending order (best-fit decreasing)
    # Keep track of original indices
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    # Each bin is a list of placed boxes: (x, y, z, w, h, d)
    bins = []
    # For each bin, track the maximum z occupied in each (x, y) column? 
    # Actually, let's use a simpler approach: for each bin, maintain a list of placed boxes
    # and check for overlap by iterating through placed boxes.
    # For efficiency with 40 boxes, this should be fine.
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in the bin without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Check overlap in all three dimensions
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        """Find the best position for a box in a given bin.
        Uses a heuristic: try to place at the lowest z, then lowest y, then lowest x.
        We'll scan through possible positions.
        For efficiency, we can use a grid-based approach or just try key positions.
        """
        # Try to find the position that minimizes (z, y, x) lexicographically
        # We'll check all possible integer positions? No, that's too slow.
        # Instead, we'll use a smarter approach: for each possible z (layer), 
        # for each possible y (shelf), find the smallest x that fits.
        
        best_pos = None
        best_key = None
        
        # For each possible z coordinate (0 to D-d)
        # But this could be many. Let's limit to "interesting" z values.
        # A common heuristic: try z = 0, and z = max_z_of_existing_boxes + d
        # Actually, let's try a subset of positions.
        
        # Simple approach: try x=0, y=0, z=0 first, then increment.
        # For 40 boxes, we can afford to check a moderate number of positions.
        
        # Let's use a different strategy: for each existing box in the bin,
        # try placing the new box adjacent to it (on top, to the right, behind).
        # Plus the origin.
        
        candidate_positions = set()
        candidate_positions.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # On top
            candidate_positions.add((px, py, pz + pd))
            # To the right
            candidate_positions.add((px + pw, py, pz))
            # Behind
            candidate_positions.add((px, py + ph, pz))
            # Diagonals
            candidate_positions.add((px + pw, py + ph, pz))
            candidate_positions.add((px + pw, py, pz + pd))
            candidate_positions.add((px, py + ph, pz + pd))
            candidate_positions.add((px + pw, py + ph, pz + pd))
        
        # Also try x=0, y=0, z=max_z
        max_z = max((pz + pd) for (px, py, pz, pw, ph, pd) in bin_boxes) if bin_boxes else 0
        candidate_positions.add((0, 0, max_z))
        max_y = max((py + ph) for (px, py, pz, pw, ph, pd) in bin_boxes) if bin_boxes else 0
        candidate_positions.add((0, max_y, 0))
        max_x = max((px + pw) for (px, py, pz, pw, ph, pd) in bin_boxes) if bin_boxes else 0
        candidate_positions.add((max_x, 0, 0))
        
        for (cx, cy, cz) in candidate_positions:
            if cx < 0 or cy < 0 or cz < 0:
                continue
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                key = (cz, cy, cx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (cx, cy, cz)
        
        return best_pos
    
    # Process boxes in sorted order
    for (orig_idx, w, h, d) in indexed_boxes:
        placed = False
        # Try to place in existing bins
        for bin_idx in range(len(bins)):
            pos = find_position(bins[bin_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[bin_idx].append((x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            # Start a new bin
            bins.append([(0, 0, 0, w, h, d)])
    
    # Build the result
    placements = []
    for bin_idx, bin_boxes in enumerate(bins):
        for (x, y, z, w, h, d) in bin_boxes:
            placements.append((bin_idx, x, y, z, w, h, d))
    
    # Reorder placements by original box index
    # We need to map back to original indices
    # Let's rebuild: for each original index, find its placement
    result = [None] * n
    for bin_idx, bin_boxes in enumerate(bins):
        for (x, y, z, w, h, d) in bin_boxes:
            # We need to know which original box this is
            # This is a problem: we lost the original index tracking
            pass
    
    # Let me redo this with proper index tracking
    bins = []  # Each bin is a list of (orig_idx, x, y, z, w, h, d)
    
    for (orig_idx, w, h, d) in indexed_boxes:
        placed = False
        for bin_idx in range(len(bins)):
            bin_boxes = bins[bin_idx]
            pos = find_position([(x, y, z, bw, bh, bd) for (_, x, y, z, bw, bh, bd) in bin_boxes], w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[bin_idx].append((orig_idx, x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            bins.append([(orig_idx, 0, 0, 0, w, h, d)])
    
    # Build final placements in original order
    placements = [None] * n
    for bin_idx, bin_boxes in enumerate(bins):
        for (orig_idx, x, y, z, w, h, d) in bin_boxes:
            placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
    
    return placements
