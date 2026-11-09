import time
import heapq

def pack(boxes, container):
    W, H, D = container
    deadline = time.time() + 140.0
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
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
        We use a set to avoid duplicates.
        """
        candidates = set()
        candidates.add((0, 0, 0))
        
        if not bin_boxes:
            return [(0, 0, 0)]
            
        x_coords = {0}
        y_coords = {0}
        z_coords = {0}
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Add current box coordinates and far faces
            x_coords.add(px)
            x_coords.add(px + pw)
            y_coords.add(py)
            y_coords.add(py + ph)
            z_coords.add(pz)
            z_coords.add(pz + pd)
            
        # Filter coords that allow the new box to fit
        x_list = [x for x in x_coords if x + w <= W]
        y_list = [y for y in y_coords if y + h <= H]
        z_list = [z for z in z_coords if z + d <= D]
        
        if not x_list or not y_list or not z_list:
            return []
            
        # Generate Cartesian product
        # To optimize, we can break early if we find a very good candidate?
        # No, we need the global min (z, y, x).
        # 40 boxes -> ~80 coords per dim -> 80^3 = 512,000 candidates.
        # This is too many to check overlap for every box in every bin.
        
        # Optimization: Only generate candidates that are "supported".
        # A position (x,y,z) is supported if:
        # 1. z == 0 (on floor)
        # 2. There is a box directly below it (z == box.z + box.d)
        # 3. There is a box directly behind it? (No, gravity is Z)
        # Actually, standard BLF assumes gravity.
        
        # Let's stick to the corner generation from the previous code but expand it slightly.
        # The previous code generated corners relative to each box.
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
        # Checking 281 candidates * 40 boxes (overlap check) = ~11k checks per bin.
        # 40 bins * 40 boxes * 11k = 17M checks. Feasible in Python? Maybe borderline.
        
        # Let's filter and sort
        valid_candidates = []
        for (x, y, z) in raw_candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                valid_candidates.append((x, y, z))
                
        # Sort by z, then y, then x (BLF)
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid_candidates

    def try_place_in_bin(bin_idx, box_w, box_h, box_d):
        bin_boxes = bins[bin_idx]
        candidates = get_candidate_positions(bin_boxes, box_w, box_h, box_d)
        
        best_pos = None
        best_score = float('inf')
        
        for (x, y, z) in candidates:
            if is_valid_placement(bin_boxes, x, y, z, box_w, box_h, box_d):
                # Score: lower z, y, x is better
                # We already sorted, so the first valid one is the best for this bin.
                # But we want the GLOBAL best across all bins.
                # So we return the best position found in this bin, and let the caller compare.
                return (x, y, z, x+y+z) # Return position and a "compactness" score
                
        return None

    placements = [None] * len(boxes)
    
    for orig_idx, w, h, d, vol in indexed_boxes:
        placed = False
        best_global_pos = None
        best_global_bin = -1
        best_global_score = float('inf')
        
        if time.time() > deadline:
            # Fallback
            b = len(bins)
            bins.append([])
            placements[orig_idx] = (b, 0, 0, 0, w, h, d)
            continue
            
        # Try existing bins
        for b in range(len(bins)):
            res = try_place_in_bin(b, w, h, d)
            if res is not None:
                x, y, z, score = res
                # We prefer lower score (more compact)
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
            # Place at origin
            bins[b].append((0, 0, 0, w, h, d))
            placements[orig_idx] = (b, 0, 0, 0, w, h, d)
            
    return placements
