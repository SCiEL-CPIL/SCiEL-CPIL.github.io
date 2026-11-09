"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import heapq

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Sort boxes by volume descending, then by max dimension descending
    # This helps fill large gaps first
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (x[4], max(x[1], x[2], x[3])), reverse=True)
    
    # Each bin stores a list of placed boxes: (x, y, z, w, h, d)
    bins = []  # List of lists of placed boxes
    
    def overlaps(b1, b2):
        """Check if two boxes overlap. b1 and b2 are (x, y, z, w, h, d)."""
        x1, y1, z1, w1, h1, d1 = b1
        x2, y2, z2, w2, h2, d2 = b2
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2 and
                z1 < z2 + d2 and z1 + d1 > z2)
    
    def is_valid_placement(bin_boxes, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in the bin and doesn't overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        
        new_box = (x, y, z, w, h, d)
        for placed in bin_boxes:
            if overlaps(new_box, placed):
                return False
        return True

    def get_candidate_positions(bin_boxes, w, h, d):
        """
        Generate candidate (x, y, z) positions.
        Strategy: Bottom-Left-Far (BLF) heuristic variant.
        Candidates are generated from the origin and the 'faces' of existing boxes.
        """
        candidates = set()
        
        # Always try the origin
        candidates.add((0, 0, 0))
        
        if not bin_boxes:
            return [(0, 0, 0)]
        
        # Generate candidates based on existing boxes
        # For each dimension, the new box can start at:
        # 1. 0 (origin)
        # 2. x + w of an existing box (right face)
        # 3. y + h of an existing box (top face)
        # 4. z + d of an existing box (far face)
        
        x_vals = {0}
        y_vals = {0}
        z_vals = {0}
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Only add if it's within bounds
            if px + pw <= W:
                x_vals.add(px + pw)
            if py + ph <= H:
                y_vals.add(py + ph)
            if pz + pd <= D:
                z_vals.add(pz + pd)
                
        # To keep it fast, we don't generate the full Cartesian product of all x, y, z values
        # if there are many boxes, but for 40 boxes it's manageable.
        # However, a smarter approach is to only consider combinations that are "supported".
        # For simplicity and robustness, we iterate through a subset of candidates.
        
        # Sort candidates to prefer lower coordinates (compactness)
        x_list = sorted([x for x in x_vals if x + w <= W])
        y_list = sorted([y for y in y_vals if y + h <= H])
        z_list = sorted([z for z in z_vals if z + d <= D])
        
        if not x_list or not y_list or not z_list:
            return []
            
        # Generate all combinations. 
        # In worst case, if we have ~40 boxes, we have ~40 unique coords per dim.
        # 40^3 = 64,000 checks per bin per box. 
        # 40 boxes * 40 bins (worst case) * 64,000 checks = 100M checks.
        # This might be too slow in Python.
        
        # Optimization: We can restrict candidates further.
        # A valid position (x,y,z) must be such that the box is "supported" in some way,
        # or we just use the BLF rule: minimize z, then y, then x.
        
        # Let's use a simpler heuristic: 
        # For each existing box, generate the 3 "corner" positions relative to it.
        # Plus origin.
        
        raw_candidates = set()
        raw_candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Positions adjacent to the existing box
            # Right
            if px + pw + w <= W:
                raw_candidates.add((px + pw, py, pz))
            # Above
            if py + ph + h <= H:
                raw_candidates.add((px, py + ph, pz))
            # Behind
            if pz + pd + d <= D:
                raw_candidates.add((px, py, pz + pd))
            
            # Corners (combinations)
            if px + pw + w <= W and py + ph + h <= H:
                raw_candidates.add((px + pw, py + ph, pz))
            if px + pw + w <= W and pz + pd + d <= D:
                raw_candidates.add((px + pw, py, pz + pd))
            if py + ph + h <= H and pz + pd + d <= D:
                raw_candidates.add((px, py + ph, pz + pd))
            
            # Full corner
            if px + pw + w <= W and py + ph + h <= H and pz + pd + d <= D:
                raw_candidates.add((px + pw, py + ph, pz + pd))
                
        # Filter and sort by "closeness" to origin (sum of coords) to prefer compactness
        valid_candidates = []
        for (x, y, z) in raw_candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                # Simple metric: prefer lower z, then y, then x
                valid_candidates.append((x, y, z))
                
        # Sort by z, then y, then x to find the "lowest" valid spot first
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        return valid_candidates

    def try_place_in_bin(bin_idx, box_w, box_h, box_d):
        """Try to place a box in a bin."""
        bin_boxes = bins[bin_idx]
        
        # Generate candidates
        candidates = get_candidate_positions(bin_boxes, box_w, box_h, box_d)
        
        for (x, y, z) in candidates:
            if is_valid_placement(bin_boxes, x, y, z, box_w, box_h, box_d):
                # Place it
                bin_boxes.append((x, y, z, box_w, box_h, box_d))
                return (x, y, z)
        
        return None
    
    # Place each box
    placements = [None] * len(boxes)
    
    for orig_idx, w, h, d, vol in indexed_boxes:
        placed = False
        
        # Check time
        if time.time() > deadline:
            # Fallback: place all remaining at origin in new bins (invalid but prevents crash)
            # Actually, we should try to just place them in a new bin if possible
            pass
            
        # Try existing bins first
        # Sort bins by current volume to try "most full" first? Or least full?
        # Usually, trying to fill existing bins is good.
        for b in range(len(bins)):
            if time.time() > deadline:
                break
            pos = try_place_in_bin(b, w, h, d)
            if pos is not None:
                placements[orig_idx] = (b, pos[0], pos[1], pos[2], w, h, d)
                placed = True
                break
        
        if not placed:
            # Open a new bin
            if time.time() <= deadline:
                b = len(bins)
                bins.append([])
                pos = try_place_in_bin(b, w, h, d)
                if pos is not None:
                    placements[orig_idx] = (b, pos[0], pos[1], pos[2], w, h, d)
                else:
                    # Should not happen if box fits in empty container
                    # Fallback
                    placements[orig_idx] = (b, 0, 0, 0, w, h, d)
            else:
                # Time up, force place in new bin
                b = len(bins)
                bins.append([])
                placements[orig_idx] = (b, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
