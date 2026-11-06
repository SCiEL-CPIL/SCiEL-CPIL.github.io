import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    # Precompute volumes
    volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(volumes)
    container_vol = W * H * D
    
    # Sort indices by volume descending for better initial packing
    order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    def get_candidates(bin_placements, w, h, d):
        """Generate candidate positions using coordinate-based approach."""
        cands = set([(0, 0, 0)])
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # For each dimension, add positions just to the right/above/behind
            cands.add((px + pw, py, pz))
            cands.add((px, py + ph, pz))
            cands.add((px, py, pz + pd))
            # Also add combinations
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            cands.add((px + pw, py + ph, pz + pd))
        return cands
    
    def is_valid_place(bin_placements, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_best_spot(bin_placements, w, h, d):
        """Find the best spot: prefer bottom-left-back (minimize z, then y, then x)."""
        cands = get_candidates(bin_placements, w, h, d)
        valid_cands = []
        for (x, y, z) in cands:
            if is_valid_place(bin_placements, x, y, z, w, h, d):
                valid_cands.append((x, y, z))
        
        if not valid_cands:
            return None
            
        valid_cands.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid_cands[0]
    
    def compute_score(placements):
        """Compute the objective: B + 0.1 * (1 - last_bin_occupied_volume / container_volume)"""
        bin_ids = set()
        for p in placements:
            bin_ids.add(p[0])
        B = len(bin_ids)
        last_bin = max(bin_ids)
        
        last_bin_vol = 0
        for p in placements:
            if p[0] == last_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        
        if container_vol == 0:
            return float(B)
            
        last_bin_fraction = last_bin_vol / container_vol
        return B + 0.1 * (1 - last_bin_fraction)
    
    def pack_initial():
        """Initial greedy packing: sort by volume desc, place in first fitting bin."""
        bins = []
        placements = [None] * n
        
        for orig_idx in order:
            w, h, d = boxes[orig_idx]
            best_bin = -1
            best_spot = None
            best_key = None
            
            for b_idx, bin_placements in enumerate(bins):
                spot = find_best_spot(bin_placements, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    # Prefer bins that are more filled (denser)
                    bin_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bin_placements)
                    # Among equal fill, prefer lower position
                    key = (-bin_vol, z, y, x)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_bin = b_idx
                        best_spot = spot
            
            if best_bin >= 0:
                x, y, z = best_spot
                bins[best_bin].append((x, y, z, w, h, d))
                placements[orig_idx] = (best_bin, x, y, z, w, h, d)
            else:
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[orig_idx] = (b_idx, 0, 0, 0, w, h, d)
        
        return bins, placements
    
    def get_bin_placements(placements, bin_id):
        return [p[1:7] for p in placements if p is not None and p[0] == bin_id]
    
    # 1. Initial Packing
    bins, placements = pack_initial()
    current_score = compute_score(placements)
    
    # 2. Local Search: try to move boxes to reduce bin count or fill last bin
    improved = True
    max_passes = 100
    pass_count = 0
    
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        if time.time() > deadline - 5:
            break
        
        last_bin_id = max(p[0] for p in placements)
        
        # A. Try to move boxes from other bins to the last bin
        for i in order:
            if time.time() > deadline - 5:
                break
            if placements[i] is None:
                continue
            orig_bin = placements[i][0]
            if orig_bin == last_bin_id:
                continue
                
            w, h, d = placements[i][4], placements[i][5], placements[i][6]
            target_placements = get_bin_placements(placements, last_bin_id)
            spot = find_best_spot(target_placements, w, h, d)
            
            if spot is not None:
                old_placement = placements[i]
                placements[i] = (last_bin_id, spot[0], spot[1], spot[2], w, h, d)
                
                new_score = compute_score(placements)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    placements[i] = old_placement
        
        # B. If last bin has only 1 box, try to move it elsewhere
        if time.time() > deadline - 5:
            break
            
        last_bin_boxes = [i for i in range(n) if placements[i] is not None and placements[i][0] == last_bin_id]
        
        if len(last_bin_boxes) == 1:
            i = last_bin_boxes[0]
            w, h, d = placements[i][4], placements[i][5], placements[i][6]
            found = False
            for j in range(n):
                if j == i or placements[j] is None:
                    continue
                target_bin_id = placements[j][0]
                if target_bin_id == last_bin_id:
                    continue
                target_placements = get_bin_placements(placements, target_bin_id)
                spot = find_best_spot(target_placements, w, h, d)
                if spot is not None:
                    placements[i] = (target_bin_id, spot[0], spot[1], spot[2], w, h, d)
                    current_score = compute_score(placements)
                    improved = True
                    found = True
                    break
        
        # C. Try 2-box swaps to increase last bin volume
        if time.time() > deadline - 10:
            break
            
        last_bin_boxes = [i for i in range(n) if placements[i] is not None and placements[i][0] == last_bin_id]
        
        for i in last_bin_boxes:
            if time.time() > deadline - 10:
                break
            w_i, h_i, d_i = placements[i][4], placements[i][5], placements[i][6]
            
            for j in range(n):
                if j == i or placements[j] is None:
                    continue
                if placements[j][0] == last_bin_id:
                    continue
                w_j, h_j, d_j = placements[j][4], placements[j][5], placements[j][6]
                
                old_i = placements[i]
                old_j = placements[j]
                
                # Remove i from last_bin
                last_bin_placements = [p for p in get_bin_placements(placements, last_bin_id) if p[0:3] != (old_i[1], old_i[2], old_i[3])]
                
                # Remove j from its bin
                j_bin_id = old_j[0]
                j_bin_placements = [p for p in get_bin_placements(placements, j_bin_id) if p[0:3] != (old_j[1], old_j[2], old_j[3])]
                
                # Try to place j in last_bin (without i)
                spot_j = find_best_spot(last_bin_placements, w_j, h_j, d_j)
                if spot_j is None:
                    continue
                
                # Try to place i in j's bin (without j)
                spot_i = find_best_spot(j_bin_placements, w_i, h_i, d_i)
                if spot_i is None:
                    continue
                
                # Apply swap
                placements[i] = (j_bin_id, spot_i[0], spot_i[1], spot_i[2], w_i, h_i, d_i)
                placements[j] = (last_bin_id, spot_j[0], spot_j[1], spot_j[2], w_j, h_j, d_j)
                
                new_score = compute_score(placements)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    placements[i] = old_i
                    placements[j] = old_j
    
    # Verification
    def is_valid(placements):
        if len(placements) != n:
            return False
        for p in placements:
            if p is None:
                return False
            b, x, y, z, w, h, d = p
            if b < 0 or x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                bi, xi, yi, zi, wi, hi, di = placements[i]
                bj, xj, yj, zj, wj, hj, dj = placements[j]
                if bi == bj:
                    if (xi < xj + wj and xi + wi > xj and
                        yi < yj + hj and yi + hi > yj and
                        zi < zj + dj and zi + di > zj):
                        return False
        return True
    
    if not is_valid(placements):
        # Fallback: Simple shelf packing to ensure validity
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    
    return placements
