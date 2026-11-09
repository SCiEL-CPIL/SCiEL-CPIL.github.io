import time
import itertools

def pack(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(0, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_volume = W * H * D
    deadline = time.time() + 135.0
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (x[4], max(x[1], x[2], x[3])), reverse=True)
    
    # Bins: list of lists of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def overlaps(b1, b2):
        x1, y1, z1, w1, h1, d1 = b1
        x2, y2, z2, w2, h2, d2 = b2
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2 and
                z1 < z2 + d2 and z1 + d1 > z2)
        
    def is_valid_placement(bin_boxes, x, y, z, w, h, d):
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
        """
        candidates = set()
        candidates.add((0, 0, 0))
        
        if not bin_boxes:
            return [(0, 0, 0)]
            
        x_coords = {0}
        y_coords = {0}
        z_coords = {0}
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            x_coords.add(px)
            x_coords.add(px + pw)
            y_coords.add(py)
            y_coords.add(py + ph)
            z_coords.add(pz)
            z_coords.add(pz + pd)
            
        x_list = [x for x in x_coords if x + w <= W]
        y_list = [y for y in y_coords if y + h <= H]
        z_list = [z for z in z_coords if z + d <= D]
        
        if not x_list or not y_list or not z_list:
            return []
            
        valid_candidates = []
        # To optimize, we can limit the number of candidates checked or break early if we find a perfect fit?
        # But we need the global min (z,y,x) for BLF.
        # 40 boxes -> ~80 coords per dim -> 80^3 = 512,000 candidates.
        # This is too many to check overlap for every box in every bin.
        
        # Optimization: Only generate candidates that are "supported".
        # A position (x,y,z) is supported if:
        # 1. z == 0 (on floor)
        # 2. There is a box directly below it (z == box.z + box.d)
        
        # Let's try to generate a smaller set:
        # For each box, the 3 faces (Right, Up, Front) and the 3 corners (RU, RF, UF) and the corner (RUF).
        # Plus origin.
        
        raw_candidates = set()
        raw_candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Faces
            if px + pw + w <= W:
                raw_candidates.add((px + pw, py, pz))
                raw_candidates.add((px + pw, py + ph, pz))
                raw_candidates.add((px + pw, py, pz + pd))
                raw_candidates.add((px + pw, py + ph, pz + pd))
                
            if py + ph + h <= H:
                raw_candidates.add((px, py + ph, pz))
                raw_candidates.add((px, py + ph, pz + pd))
                
            if pz + pd + d <= D:
                raw_candidates.add((px, py, pz + pd))
                
        # This set is roughly 1 + 40 * 7 = 281 candidates.
        
        # Let's filter and sort
        valid_candidates = []
        for (x, y, z) in raw_candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if is_valid_placement(bin_boxes, x, y, z, w, h, d):
                    valid_candidates.append((x, y, z))
                    
        # Sort by z, then y, then x (BLF)
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid_candidates

    def try_place_in_bin(bin_idx, box_w, box_h, box_d):
        bin_boxes = bins[bin_idx]
        candidates = get_candidate_positions(bin_boxes, box_w, box_h, box_d)
        
        if not candidates:
            return None
            
        # Return the best (lowest z,y,x) valid position
        x, y, z = candidates[0]
        return (x, y, z, x + y + z)
        
    placements = [None] * len(boxes)
    
    for orig_idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place in a new bin at origin
            b = len(bins)
            bins.append([])
            bins[b].append((0, 0, 0, w, h, d))
            placements[orig_idx] = (b, 0, 0, 0, w, h, d)
            continue
            
        best_global_pos = None
        best_global_bin = -1
        best_global_score = float('inf')
        
        # Try existing bins
        for b in range(len(bins)):
            res = try_place_in_bin(b, w, h, d)
            if res is not None:
                x, y, z, score = res
                if score < best_global_score:
                    best_global_score = score
                    best_global_pos = (x, y, z)
                    best_global_bin = b
                    
        if best_global_bin != -1:
            bins[best_global_bin].append((best_global_pos[0], best_global_pos[1], best_global_pos[2], w, h, d))
            placements[orig_idx] = (best_global_bin, best_global_pos[0], best_global_pos[1], best_global_pos[2], w, h, d)
        else:
            # Open new bin
            b = len(bins)
            bins.append([])
            bins[b].append((0, 0, 0, w, h, d))
            placements[orig_idx] = (b, 0, 0, 0, w, h, d)
            
    # Renumber bins to 0..B-1 such that the fullest bin has the highest label.
    # This maximizes the "last-bin occupied volume fraction" term in the score.
    
    if not bins:
        return []
        
    bin_volumes = [sum(w * h * d for (_, _, _, w, h, d) in bin) for bin in bins]
    
    # Sort bin indices by volume descending
    # The bin with the largest volume should get the highest label (len(bins)-1).
    # The bin with the second largest volume should get the second highest label (len(bins)-2).
    # ...
    # The bin with the smallest volume should get label 0.
    
    sorted_bin_indices = sorted(range(len(bins)), key=lambda i: bin_volumes[i], reverse=True)
    
    old_to_new = {}
    for k, old_idx in enumerate(sorted_bin_indices):
        new_idx = len(bins) - 1 - k
        old_to_new[old_idx] = new_idx
        
    final_placements = []
    for b, x, y, z, w, h, d in placements:
        new_b = old_to_new[b]
        final_placements.append((new_b, x, y, z, w, h, d))
        
    return final_placements
