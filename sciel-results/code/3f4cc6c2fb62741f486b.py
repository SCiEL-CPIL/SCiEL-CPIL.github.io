"""Strong 3D Container Loading: multi-strategy with skyline and best-fit decreasing."""
import time
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin vs 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_volume = W * H * D
    
    # Precompute orientations for each box
    box_data = []
    for idx, (w, h, d) in enumerate(boxes):
        # Generate all unique orientations
        seen = set()
        orients = []
        for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        
        # Filter to only orientations that fit in container
        valid_orients = [(bw, bh, bd) for bw, bh, bd in orients if bw <= W and bh <= H and bd <= D]
        if not valid_orients:
            valid_orients = [(w, h, d)]
        
        # Sort orientations: prefer those with smaller max dimension (more "flat")
        valid_orients.sort(key=lambda o: (max(o), o[0], o[1], o[2]))
        
        volume = w * h * d
        box_data.append((idx, valid_orients, volume, w, h, d))
    
    # Multiple sorting strategies
    box_data_vol = sorted(box_data, key=lambda x: x[2], reverse=True)
    box_data_maxdim = sorted(box_data, key=lambda x: max(x[3], x[4], x[5]), reverse=True)
    # Cubic-first: prefer boxes with more uniform dimensions
    box_data_cubic = sorted(box_data, key=lambda x: (max(x[3], x[4], x[5]) / max(1, min(x[3], x[4], x[5])), x[2]), reverse=True)
    
    def try_place(boxes_to_place, W, H, D, deadline, candidate_order='zyx'):
        """Try to place boxes using a candidate position approach with best-fit scoring."""
        placed = []  # list of (x, y, z, bw, bh, bd)
        results = [None] * len(boxes_to_place)
        
        def get_candidates():
            """Generate candidate positions from placed boxes and origin."""
            cx_set = {0}
            cy_set = {0}
            cz_set = {0}
            for (px, py, pz, pw, ph, pd) in placed:
                cx_set.add(px)
                cx_set.add(px + pw)
                cy_set.add(py)
                cy_set.add(py + ph)
                cz_set.add(pz)
                cz_set.add(pz + pd)
            cx_list = sorted(cx_set)
            cy_list = sorted(cy_set)
            cz_list = sorted(cz_set)
            return cx_list, cy_list, cz_list
        
        total_placed_volume = 0
        
        for idx, orients, volume, ow, oh, od in boxes_to_place:
            if time.time() > deadline:
                # Time out - mark remaining as not placed
                for j in range(idx, len(boxes_to_place)):
                    if results[j] is None:
                        oj, oorient, ov, ow2, oh2, od2 = boxes_to_place[j]
                        results[oj] = (-1, 0, 0, 0, ow2, oh2, od2)
                # Fill any that are still None
                for i in range(len(boxes_to_place)):
                    if results[i] is None:
                        w_i, h_i, d_i = boxes_to_place[i][3], boxes_to_place[i][4], boxes_to_place[i][5]
                        results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
                break
            
            # Get candidate positions
            cx_list, cy_list, cz_list = get_candidates()
            
            found = False
            
            # Generate all valid (position, orientation) pairs with best-fit score
            valid_placements = []
            
            for cx in cx_list:
                for cy in cy_list:
                    for cz in cz_list:
                        for (bw, bh, bd) in orients:
                            # Check bounds
                            if cx + bw > W or cy + bh > H or cz + bd > D:
                                continue
                            
                            # Check overlap with all placed boxes
                            overlap = False
                            for (px, py, pz, pw, ph, pd) in placed:
                                # No overlap if separated in any dimension
                                if (cx + bw <= px) or (px + pw <= cx) or \
                                   (cy + bh <= py) or (py + ph <= cy) or \
                                   (cz + bd <= pz) or (pz + pd <= cz):
                                    continue
                                # Overlap
                                overlap = True
                                break
                            
                            if overlap:
                                continue
                            
                            # Compute best-fit score: prefer positions that are tight against boundaries/boxes
                            # Score = sum of distances to nearest boundary/box face in each axis
                            # Lower score = tighter fit = better
                            score = 0.0
                            
                            # X-axis: distance to left wall (cx) + distance to right wall (W - cx - bw)
                            # Prefer being close to left wall
                            score_x = cx + (W - cx - bw)
                            score += score_x
                            
                            # Y-axis
                            score_y = cy + (H - cy - bh)
                            score += score_y
                            
                            # Z-axis
                            score_z = cz + (D - cz - bd)
                            score += score_z
                            
                            # Also consider contact with other boxes (tightness)
                            # For each placed box, compute minimum gap in each axis
                            for (px, py, pz, pw, ph, pd) in placed:
                                # X gap
                                if cx + bw > px and px + pw > cx:
                                    # Overlapping in X, check Y and Z
                                    pass
                                else:
                                    # Gap in X
                                    if cx >= px + pw:
                                        gap_x = cx - (px + pw)
                                    else:
                                        gap_x = px - (cx + bw)
                                    # Only count if not overlapping in Y and Z
                                    if (cy + bh <= py or py + ph <= cy or 
                                        cz + bd <= pz or pz + pd <= cz):
                                        pass  # Already separated in Y or Z, don't double count
                            
                            valid_placements.append((score, cx, cy, cz, bw, bh, bd))
            
            # Sort by score (lower is better), then by position preference
            if candidate_order == 'zyx':
                valid_placements.sort(key=lambda p: (p[0], p[3], p[2], p[1]))
            elif candidate_order == 'xyz':
                valid_placements.sort(key=lambda p: (p[0], p[1], p[2], p[3]))
            else:
                valid_placements.sort(key=lambda p: (p[0], p[3], p[2], p[1]))
            
            # Try placements in order of best score
            for (score, cx, cy, cz, bw, bh, bd) in valid_placements:
                # Place this box
                placed.append((cx, cy, cz, bw, bh, bd))
                results[idx] = (0, cx, cy, cz, bw, bh, bd)
                total_placed_volume += volume
                found = True
                break
            
            if not found:
                results[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        # Fill any remaining None entries
        for i in range(len(boxes_to_place)):
            if results[i] is None:
                w_i, h_i, d_i = boxes_to_place[i][3], boxes_to_place[i][4], boxes_to_place[i][5]
                results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
        
        return results, total_placed_volume
    
    # Run multiple strategies
    all_results = []
    
    # Strategy 1: Volume-descending, ZYX ordering
    results1, vol1 = try_place(box_data_vol, W, H, D, deadline, 'zyx')
    all_results.append((vol1, results1))
    
    # Strategy 2: Max-dimension descending, ZYX ordering
    if time.time() < deadline:
        results2, vol2 = try_place(box_data_maxdim, W, H, D, deadline, 'zyx')
        all_results.append((vol2, results2))
    
    # Strategy 3: Volume-descending, XYZ ordering
    if time.time() < deadline:
        results3, vol3 = try_place(box_data_vol, W, H, D, deadline, 'xyz')
        all_results.append((vol3, results3))
    
    # Strategy 4: Cubic-first, ZYX ordering
    if time.time() < deadline:
        results4, vol4 = try_place(box_data_cubic, W, H, D, deadline, 'zyx')
        all_results.append((vol4, results4))
    
    # Choose the best result based on placed volume
    best_results = max(all_results, key=lambda x: x[0])
    
    return best_results[1]
# EVOLVE-BLOCK-END
