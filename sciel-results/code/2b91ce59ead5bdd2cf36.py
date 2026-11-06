import time
import numpy as np
from collections import defaultdict

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    # Sort boxes by volume descending for initial packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Precompute container volume
    container_vol = W * H * D if W > 0 and H > 0 and D > 0 else 1
    
    def get_candidates(bin_placements):
        """Generate candidate positions for a new box using face-adjacent positions."""
        cands = set([(0, 0, 0)])
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Positions adjacent to each face
            cands.add((px + pw, py, pz))
            cands.add((px, py + ph, pz))
            cands.add((px, py, pz + pd))
            # Corner combinations
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            cands.add((px + pw, py + ph, pz + pd))
        return cands
    
    def is_valid_place(bin_placements, x, y, z, w, h, d):
        """Check if a box can be placed at (x,y,z) without overlap and within bounds."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Check overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_best_spot(bin_placements, w, h, d):
        """Find the best spot in a bin using Bottom-Left-Back heuristic."""
        cands = get_candidates(bin_placements)
        valid_cands = []
        for (x, y, z) in cands:
            if is_valid_place(bin_placements, x, y, z, w, h, d):
                valid_cands.append((x, y, z))
        
        if not valid_cands:
            return None
            
        # Sort by z (back), then y (left), then x (bottom) - prefer lower/back positions
        valid_cands.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid_cands[0]
    
    def pack_initial():
        """Initial packing using best-fit decreasing with B-L-B heuristic."""
        bins = []
        placements = [None] * n
        
        for idx, (orig_idx, w, h, d, vol) in enumerate(indexed_boxes):
            best_bin = -1
            best_spot = None
            best_key = None
            
            for b_idx, bin_placements in enumerate(bins):
                spot = find_best_spot(bin_placements, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    # Prefer bins with more volume already used, then lower position
                    bin_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bin_placements)
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
    
    def score_placements(placements):
        """Calculate the score: B + 0.1*(1 - last_bin_vol/container_vol)."""
        if not placements:
            return 0.0
        bin_ids = set(p[0] for p in placements)
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
    
    def get_bin_placements(placements, bin_id):
        """Get all placements in a specific bin."""
        return [p[1:7] for p in placements if p is not None and p[0] == bin_id]
    
    def is_valid(placements):
        """Verify the placement is valid (no overlaps, within bounds)."""
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
    
    # 1. Initial Packing
    bins, placements = pack_initial()
    current_score = score_placements(placements)
    
    # 2. Local Search / Improvement
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
        for i in [x[0] for x in indexed_boxes]:
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
                
                new_score = score_placements(placements)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    placements[i] = old_placement
        
        # B. Try to empty the last bin by moving its boxes to other bins
        if time.time() > deadline - 5:
            break
            
        last_bin_boxes = [i for i in range(n) if placements[i] is not None and placements[i][0] == last_bin_id]
        
        # Try to move boxes out of last bin to reduce its count
        for i in last_bin_boxes[:]:  # Copy to allow modification
            if time.time() > deadline - 5:
                break
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
                    current_score = score_placements(placements)
                    improved = True
                    found = True
                    break
        
        # C. Try 2-box swaps to increase last bin volume
        if time.time() > deadline - 8:
            break
            
        last_bin_boxes = [i for i in range(n) if placements[i] is not None and placements[i][0] == last_bin_id]
        
        for i in last_bin_boxes:
            if time.time() > deadline - 8:
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
                last_bin_placements = [p for p in get_bin_placements(placements, last_bin_id) 
                                       if (p[0], p[1], p[2]) != (old_i[1], old_i[2], old_i[3])]
                
                # Remove j from its bin
                j_bin_id = old_j[0]
                j_bin_placements = [p for p in get_bin_placements(placements, j_bin_id) 
                                    if (p[0], p[1], p[2]) != (old_j[1], old_j[2], old_j[3])]
                
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
                
                new_score = score_placements(placements)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    placements[i] = old_i
                    placements[j] = old_j
        
        # D. Try to merge bins: move all boxes from bin X to bin Y if possible
        if time.time() > deadline - 10:
            break
            
        bin_ids = sorted(set(p[0] for p in placements if p is not None))
        if len(bin_ids) > 1:
            # Try to merge each bin into the last bin
            for b_idx in bin_ids[:-1]:  # Skip last bin
                if time.time() > deadline - 10:
                    break
                boxes_in_bin = [i for i in range(n) if placements[i] is not None and placements[i][0] == b_idx]
                if not boxes_in_bin:
                    continue
                
                # Check if all boxes can fit in last bin
                can_merge = True
                last_placements = get_bin_placements(placements, last_bin_id)
                for i in boxes_in_bin:
                    w, h, d = placements[i][4], placements[i][5], placements[i][6]
                    if find_best_spot(last_placements, w, h, d) is None:
                        can_merge = False
                        break
                
                if can_merge:
                    # Perform the merge
                    for i in boxes_in_bin:
                        w, h, d = placements[i][4], placements[i][5], placements[i][6]
                        spot = find_best_spot(get_bin_placements(placements, last_bin_id), w, h, d)
                        if spot is not None:
                            placements[i] = (last_bin_id, spot[0], spot[1], spot[2], w, h, d)
                    current_score = score_placements(placements)
                    improved = True
    
    # Final verification and fallback
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
