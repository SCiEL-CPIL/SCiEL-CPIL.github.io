import time
import itertools

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 115.0  # safety margin (150s limit, 35s buffer)
    
    # Generate all 6 unique rotations for each box (dedup if symmetric)
    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    box_rots = []
    for i, (w, h, d) in enumerate(boxes):
        box_rots.append(get_rotations(w, h, d))
    
    # Sorting strategies
    def sort_volume_desc(idx_list):
        return sorted(idx_list, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    def sort_max_dim_desc(idx_list):
        return sorted(idx_list, key=lambda i: -max(boxes[i]))
    
    def sort_min_dim_asc(idx_list):
        return sorted(idx_list, key=lambda i: min(boxes[i]))
    
    def sort_ratio_desc(idx_list):
        return sorted(idx_list, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]) / max(1, max(boxes[i])**2))
    
    def sort_by_fit(idx_list):
        min_c = min(W, H, D)
        return sorted(idx_list, key=lambda i: min(boxes[i]) / max(1, min_c))
    
    def sort_cubic_first(idx_list):
        # Prefer boxes that are more cube-like (min dim close to max dim)
        def cube_ratio(i):
            b = boxes[i]
            mn, mx = min(b), max(b)
            return mn / mx if mx > 0 else 0
        return sorted(idx_list, key=lambda i: -cube_ratio(i))
    
    strategies = [sort_volume_desc, sort_max_dim_desc, sort_min_dim_asc, sort_ratio_desc, sort_by_fit, sort_cubic_first]
    
    def check_overlap(x, y, z, rw, rh, rd, placed_rects):
        for (px, py, pz, pw, ph, pd) in placed_rects:
            if (x < px + pw and x + rw > px and
                y < py + ph and y + rh > py and
                z < pz + pd and z + rd > pz):
                return True
        return False
    
    def place_box_3d(rw, rh, rd, placed_rects):
        """
        Try to place a box with given dimensions in the container using Box-Tree (maximal void) approach.
        Generates candidate positions from the corners of maximal empty rectangles.
        Returns (x, y, z) or None.
        """
        # Generate candidate positions from existing box edges + origin
        x_candidates = {0}
        y_candidates = {0}
        z_candidates = {0}
        
        for (px, py, pz, pw, ph, pd) in placed_rects:
            x_candidates.add(px)
            x_candidates.add(px + pw)
            y_candidates.add(py)
            y_candidates.add(py + ph)
            z_candidates.add(pz)
            z_candidates.add(pz + pd)
        
        # Filter to valid range
        x_candidates = sorted(x for x in x_candidates if 0 <= x <= W - rw)
        y_candidates = sorted(y for y in y_candidates if 0 <= y <= H - rh)
        z_candidates = sorted(z for z in z_candidates if 0 <= z <= D - rd)
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None
        
        # Limit candidates to keep it fast
        max_cands = 25
        if len(x_candidates) > max_cands:
            # Keep candidates closest to existing boxes (better fit)
            x_candidates = x_candidates[:max_cands]
        if len(y_candidates) > max_cands:
            y_candidates = y_candidates[:max_cands]
        if len(z_candidates) > max_cands:
            z_candidates = z_candidates[:max_cands]
        
        # Try all combinations - prioritize positions that maximize contact with existing boxes
        best_pos = None
        best_contact = -1
        
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if check_overlap(x, y, z, rw, rh, rd, placed_rects):
                        continue
                    
                    # Score by contact (boxes touching boundaries or other boxes are better)
                    contact = 0
                    # Contact with container boundaries
                    if x == 0: contact += 1
                    if x + rw == W: contact += 1
                    if y == 0: contact += 1
                    if y + rh == H: contact += 1
                    if z == 0: contact += 1
                    if z + rd == D: contact += 1
                    
                    # Contact with existing boxes
                    for (px, py, pz, pw, ph, pd) in placed_rects:
                        if (abs(x - (px+pw)) < 1e-9 or abs(x + rw - px) < 1e-9):
                            contact += 0.1
                        if (abs(y - (py+ph)) < 1e-9 or abs(y + rh - py) < 1e-9):
                            contact += 0.1
                        if (abs(z - (pz+pd)) < 1e-9 or abs(z + rd - pz) < 1e-9):
                            contact += 0.1
                    
                    if contact > best_contact:
                        best_contact = contact
                        best_pos = (x, y, z)
                    
                    # Early termination if we found a position touching 3+ boundaries
                    if contact >= 3:
                        break
                else:
                    continue
                break
            else:
                continue
            break
        
        # If no high-contact position, just return the first valid one
        if best_pos is None:
            for x in x_candidates:
                for y in y_candidates:
                    for z in z_candidates:
                        if not check_overlap(x, y, z, rw, rh, rd, placed_rects):
                            return (x, y, z)
            return None
        
        return best_pos
    
    def run_strategy(sort_func):
        """Run a packing strategy and return placements."""
        placed_rects = []
        result = [None] * n
        order = sort_func(list(range(n)))
        
        for i in order:
            if time.time() > deadline:
                break
            
            # Try each rotation
            best_pos = None
            best_rot_vol = -1
            for rw, rh, rd in box_rots[i]:
                if rw > W or rh > H or rd > D:
                    continue
                pos = place_box_3d(rw, rh, rd, placed_rects)
                if pos is not None:
                    vol = rw * rh * rd
                    # Prefer rotation that maximizes volume in the chosen orientation
                    if vol > best_rot_vol:
                        best_rot_vol = vol
                        best_pos = (pos, rw, rh, rd)
            
            if best_pos is not None:
                (x, y, z), rw, rh, rd = best_pos
                placed_rects.append((x, y, z, rw, rh, rd))
                result[i] = (0, x, y, z, rw, rh, rd)
            else:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        # Fill in any that weren't processed due to time
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return result
    
    def run_layer_packing(sort_func, num_layers):
        """
        Divide container into horizontal layers and pack each layer independently.
        """
        result = [None] * n
        order = sort_func(list(range(n)))
        layer_height = H / num_layers if num_layers > 0 else H
        
        for layer_idx in range(num_layers):
            z_start = layer_idx * layer_height
            z_end = min(z_start + layer_height, H)
            layer_h = z_end - z_start
            
            if layer_h <= 0:
                continue
            
            placed_2d = []
            
            for i in order:
                if time.time() > deadline:
                    break
                
                bw, bh, bd = boxes[i]
                
                valid_rots = []
                for rw, rh, rd in box_rots[i]:
                    if rh <= layer_h + 1e-9:
                        valid_rots.append((rw, rd))
                        break
                
                if not valid_rots:
                    continue
                
                rw, rd = valid_rots[0]
                
                x_candidates = {0}
                y_candidates = {0}
                
                for (px, py, pw, pd) in placed_2d:
                    x_candidates.add(px)
                    x_candidates.add(px + pw)
                    y_candidates.add(py)
                    y_candidates.add(py + pd)
                
                x_candidates = sorted(x for x in x_candidates if 0 <= x <= W - rw)
                y_candidates = sorted(y for y in y_candidates if 0 <= y <= D - rd)
                
                max_cands = 15
                if len(x_candidates) > max_cands:
                    x_candidates = x_candidates[:max_cands]
                if len(y_candidates) > max_cands:
                    y_candidates = y_candidates[:max_cands]
                
                placed = False
                for x in x_candidates:
                    for y in y_candidates:
                        overlap = False
                        for (px, py, pw, pd) in placed_2d:
                            if (x < px + pw and x + rw > px and
                                y < py + pd and y + rd > py):
                                overlap = True
                                break
                        
                        if not overlap:
                            placed_2d.append((x, y, rw, rd))
                            result[i] = (0, x, y, z_start, rw, bh, rd)
                            placed = True
                            break
                    if placed:
                        break
        
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return result
    
    best_placements = None
    best_score = -1
    
    # Try all strategies with full 3D packing
    for sort_func in strategies:
        if time.time() > deadline:
            break
        
        placements = run_strategy(sort_func)
        
        total_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        score = total_vol / (W * H * D) if W * H * D > 0 else 0
        
        if score > best_score:
            best_score = score
            best_placements = placements
    
    # Try layer decomposition with different numbers of layers
    if time.time() < deadline:
        for num_layers in [2, 3, 4, 5]:
            if time.time() > deadline:
                break
            
            for sort_func in strategies[:3]:  # Use fewer strategies for layers to save time
                if time.time() > deadline:
                    break
                
                placements = run_layer_packing(sort_func, num_layers)
                
                total_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
                score = total_vol / (W * H * D) if W * H * D > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_placements = placements
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
