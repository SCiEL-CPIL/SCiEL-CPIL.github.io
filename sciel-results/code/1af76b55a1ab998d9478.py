"""3D Container Loading: BFDH with skyline-based candidate generation and multi-strategy."""
import time
import math

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    t_start = time.time()
    deadline = t_start + 105.0  # safety margin vs 120s limit
    
    # Precompute orientations for each box
    box_data = []
    for idx, (w, h, d) in enumerate(boxes):
        seen = set()
        orients = []
        for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        
        valid_orients = [(bw, bh, bd) for bw, bh, bd in orients if bw <= W and bh <= H and bd <= D]
        if not valid_orients:
            valid_orients = [(w, h, d)]
        
        valid_orients.sort(key=lambda o: (max(o), o[0], o[1], o[2]))
        
        volume = w * h * d
        box_data.append((idx, valid_orients, volume, w, h, d))
    
    # Sort by volume descending (Best-Fit Decreasing)
    box_data_vol = sorted(box_data, key=lambda x: x[2], reverse=True)
    
    def execute_placement(box_order):
        """Execute a placement strategy with a specific box ordering using skyline candidates."""
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * n
        
        def check_no_overlap(cx, cy, cz, bw, bh, bd):
            """Check if a box at (cx,cy,cz) with dims (bw,bh,bd) doesn't overlap any placed box."""
            for (px, py, pz, pw, ph, pd) in placed:
                if cx + bw <= px or px + pw <= cx:
                    continue
                if cy + bh <= py or py + ph <= cy:
                    continue
                if cz + bd <= pz or pz + pd <= cz:
                    continue
                return False
            return True
        
        def get_skyline_candidates():
            """
            Generate candidate positions using a skyline approach.
            We consider all combinations of x, y, z coordinates that are either 0
            or the right/top/back face of a placed box, but we use a smarter
            filtering to reduce the candidate set.
            """
            cx_set = {0}
            cy_set = {0}
            cz_set = {0}
            for (px, py, pz, pw, ph, pd) in placed:
                if px + pw <= W:
                    cx_set.add(px + pw)
                if py + ph <= H:
                    cy_set.add(py + ph)
                if pz + pd <= D:
                    cz_set.add(pz + pd)
            
            cx_list = sorted(cx_set)
            cy_list = sorted(cy_set)
            cz_list = sorted(cz_set)
            
            return cx_list, cy_list, cz_list
        
        def try_place_box(idx, orients, volume, ow, oh, od):
            """Try to place a box using best-fit with skyline candidates."""
            cx_list, cy_list, cz_list = get_skyline_candidates()
            
            # Generate all candidate positions
            candidates = []
            for cz in cz_list:
                for cy in cy_list:
                    for cx in cx_list:
                        candidates.append((cx, cy, cz))
            
            # Sort by (z, y, x) to prefer lowest, then most-left, then most-front
            candidates.sort(key=lambda p: (p[2], p[1], p[0]))
            
            # Precompute min dimensions for early pruning
            min_w = min(o[0] for o in orients)
            min_h = min(o[1] for o in orients)
            min_d = min(o[2] for o in orients)
            
            best_pos = None
            best_key = None
            
            for (cx, cy, cz) in candidates:
                # Early pruning
                if cx + min_w > W or cy + min_h > H or cz + min_d > D:
                    continue
                
                # Try each orientation
                for (bw, bh, bd) in orients:
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    if not check_no_overlap(cx, cy, cz, bw, bh, bd):
                        continue
                    
                    # Found a valid position - compute contact score
                    contact_score = 0
                    if cz == 0:
                        contact_score += 1000
                    if cy == 0:
                        contact_score += 500
                    if cx == 0:
                        contact_score += 250
                    
                    # Check adjacency with placed boxes
                    for (px, py, pz, pw, ph, pd) in placed:
                        # Adjacent in x
                        if abs((cx + bw) - px) < 1e-9 or abs((px + pw) - cx) < 1e-9:
                            if not (cy + bh <= py or py + ph <= cy) and not (cz + bd <= pz or pz + pd <= cz):
                                contact_score += 50
                        # Adjacent in y
                        if abs((cy + bh) - py) < 1e-9 or abs((py + ph) - cy) < 1e-9:
                            if not (cx + bw <= px or px + pw <= cx) and not (cz + bd <= pz or pz + pd <= cz):
                                contact_score += 30
                        # Adjacent in z
                        if abs((cz + bd) - pz) < 1e-9 or abs((pz + pd) - cz) < 1e-9:
                            if not (cx + bw <= px or px + pw <= cx) and not (cy + bh <= py or py + ph <= cy):
                                contact_score += 10
                    
                    key = (cz, -contact_score, cy, cx)
                    
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = (cx, cy, cz, bw, bh, bd)
                    break  # No need to try other orientations for this position
                
                # Early termination if we have a good position
                if best_pos is not None and best_key[0] == 0 and best_key[1] >= 1000:
                    break
            
            return best_pos
        
        # Process boxes in the given order
        for idx, orients, volume, ow, oh, od in box_order:
            if time.time() > deadline:
                break
            
            best_pos = try_place_box(idx, orients, volume, ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz, bw, bh, bd = best_pos
                placed.append((cx, cy, cz, bw, bh, bd))
                results[idx] = (0, cx, cy, cz, bw, bh, bd)
            else:
                results[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        # Fill in any remaining None
        for i in range(n):
            if results[i] is None:
                results[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        # Calculate filled volume
        filled = 0
        for r in results:
            if r[0] == 0:
                filled += r[4] * r[5] * r[6]
        
        return results, filled
    
    # Strategy 1: Volume descending (BFD)
    results1, filled1 = execute_placement(box_data_vol)
    
    # Strategy 2: Sort by max dimension descending, then volume
    box_data_maxdim = sorted(box_data, key=lambda x: (max(x[3], x[4], x[5]), x[2]), reverse=True)
    results2, filled2 = execute_placement(box_data_maxdim)
    
    # Strategy 3: Sort by aspect ratio (more cubic boxes first), then volume
    def aspect_ratio(box):
        w, h, d = box[3], box[4], box[5]
        dims = sorted([w, h, d])
        return dims[2] / max(dims[0], 1)
    
    box_data_cubic = sorted(box_data, key=lambda x: (aspect_ratio(x), x[2]), reverse=True)
    results3, filled3 = execute_placement(box_data_cubic)
    
    # Choose the best result
    best_results = results1
    best_filled = filled1
    if filled2 > best_filled:
        best_results = results2
        best_filled = filled2
    if filled3 > best_filled:
        best_results = results3
        best_filled = filled3
    
    return best_results
