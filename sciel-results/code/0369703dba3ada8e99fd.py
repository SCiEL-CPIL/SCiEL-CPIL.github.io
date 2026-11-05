import time
import sys
import numpy as np

def _box_volume(box):
    w, h, d = box
    return w * h * d

def _can_place(box, pos, placements, W, H, D):
    x, y, z = pos
    w, h, d = box
    
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    
    for (px, py, pz, pw, ph, pd) in placements:
        if x < px + pw and px < x + w and \
           y < py + ph and py < y + h and \
           z < pz + pd and pz < z + d:
            return False
    return True

def _find_position_bfd(box, placements, W, H, D):
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
        
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
        
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D):
                    return (x, y, z)
                    
    return None

def _find_position_best_fit(box, placements, W, H, D):
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (px, py, pz, pw, ph, pd) in placements:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
        
    valid_xs = sorted([x for x in xs if x + w <= W])
    valid_ys = sorted([y for y in ys if y + h <= H])
    valid_zs = sorted([z for z in zs if z + d <= D])
    
    if not valid_xs or not valid_ys or not valid_zs:
        return None
        
    best_pos = None
    best_score = float('inf')
    
    if not placements:
        return (0, 0, 0)
        
    for z in valid_zs:
        for y in valid_ys:
            for x in valid_xs:
                if _can_place(box, (x, y, z), placements, W, H, D):
                    min_dist = float('inf')
                    for (px, py, pz, pw, ph, pd) in placements:
                        dx = max(0, max(x, px) - min(x + w, px + pw))
                        dy = max(0, max(y, py) - min(y + h, py + ph))
                        dz = max(0, max(z, pz) - min(z + d, pz + pd))
                        dist = dx + dy + dz
                        if dist < min_dist:
                            min_dist = dist
                            if min_dist == 0:
                                break
                    score = z * 10000 + y * 100 + x + min_dist * 0.1
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
                        
    return best_pos

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = len(boxes)
    if n == 0:
        return []
    
    def _pack_with_order(order, finder_func):
        bins = []
        result = [None] * n
        
        for orig_idx in order:
            box = boxes[orig_idx]
            w, h, d = box
            placed = False
            
            for bi in range(len(bins)):
                pos = finder_func(box, bins[bi], W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((x, y, z, w, h, d))
                    result[orig_idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                result[orig_idx] = (bi, 0, 0, 0, w, h, d)
        
        for i in range(n):
            if result[i] is None:
                bi = len(bins)
                w, h, d = boxes[i]
                bins.append([(0, 0, 0, w, h, d)])
                result[i] = (bi, 0, 0, 0, w, h, d)
        
        return bins, result
    
    def _score_solution(bins, result):
        if not bins:
            return float('inf')
        
        occupied_bins = set()
        for r in result:
            occupied_bins.add(r[0])
        B = len(occupied_bins)
        
        last_bin_id = max(r[0] for r in result)
        last_vol = sum(p[3] * p[4] * p[5] for p in bins[last_bin_id])
        container_vol = W * H * D
        fill = last_vol / container_vol if container_vol > 0 else 0.0
        
        return B + 0.1 * (1 - fill)
    
    # Strategy 1: Volume descending, then max dimension descending (BFL)
    order1 = sorted(range(n), key=lambda i: (_box_volume(boxes[i]), max(boxes[i])), reverse=True)
    
    # Strategy 2: Max dimension descending, then volume descending (BFL)
    order2 = sorted(range(n), key=lambda i: (max(boxes[i]), _box_volume(boxes[i])), reverse=True)
    
    # Strategy 3: Volume descending (Best Fit)
    order3 = sorted(range(n), key=lambda i: (_box_volume(boxes[i]), max(boxes[i])), reverse=True)
    
    # Strategy 4: Max dimension descending (Best Fit)
    order4 = sorted(range(n), key=lambda i: (max(boxes[i]), _box_volume(boxes[i])), reverse=True)
    
    # Strategy 5: Aspect ratio sorting
    order5 = sorted(range(n), key=lambda i: (min(boxes[i]) / max(boxes[i]), _box_volume(boxes[i])), reverse=True)
    
    # Strategy 6: Volume / max_dim ratio
    order6 = sorted(range(n), key=lambda i: (_box_volume(boxes[i]) / max(boxes[i]), _box_volume(boxes[i])), reverse=True)
    
    best_score = float('inf')
    best_bins = None
    best_result = None
    
    strategies = [
        (order1, _find_position_bfd),
        (order2, _find_position_bfd),
        (order3, _find_position_best_fit),
        (order4, _find_position_best_fit),
        (order5, _find_position_bfd),
        (order6, _find_position_bfd),
    ]
    
    for order, finder in strategies:
        if time.time() > deadline - 30.0:
            break
        
        bins_try, result_try = _pack_with_order(order, finder)
        score_try = _score_solution(bins_try, result_try)
        
        if score_try < best_score:
            best_score = score_try
            best_bins = bins_try
            best_result = result_try
    
    bins = best_bins
    result = best_result
            
    # Phase 2: Optimize Last Bin Fill
    if time.time() < deadline - 3.0:
        last_bin_id = max(r[0] for r in result)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # --- Attempt Moves ---
            for idx in range(n):
                if time.time() > deadline - 3.0:
                    break
                if result[idx][0] == last_bin_id:
                    continue
                    
                cur_bin_id = result[idx][0]
                box = boxes[idx]
                w, h, d = box
                
                last_placements = bins[last_bin_id]
                pos = _find_position_bfd(box, last_placements, W, H, D)
                
                if pos is not None:
                    cur_placements = bins[cur_bin_id]
                    target_placement = (result[idx][1], result[idx][2], result[idx][3], w, h, d)
                    
                    removed = False
                    for j in range(len(cur_placements)):
                        if cur_placements[j] == target_placement:
                            cur_placements.pop(j)
                            removed = True
                            break
                            
                    if removed:
                        bins[last_bin_id].append((pos[0], pos[1], pos[2], w, h, d))
                        result[idx] = (last_bin_id, pos[0], pos[1], pos[2], w, h, d)
                        improved = True
                        break
                        
            # --- Attempt Swaps ---
            if not improved and time.time() < deadline - 10.0:
                last_boxes_indices = [i for i in range(n) if result[i][0] == last_bin_id]
                
                for idx_last in last_boxes_indices:
                    if time.time() > deadline - 10.0:
                        break
                    
                    box_last = boxes[idx_last]
                    wl, hl, dl = box_last
                    pos_last = (result[idx_last][1], result[idx_last][2], result[idx_last][3])
                    target_last_placement = (pos_last[0], pos_last[1], pos_last[2], wl, hl, dl)
                    
                    for idx_other in range(n):
                        if time.time() > deadline - 10.0:
                            break
                        if result[idx_other][0] == last_bin_id:
                            continue
                            
                        box_other = boxes[idx_other]
                        wo, ho, do = box_other
                        pos_other = (result[idx_other][1], result[idx_other][2], result[idx_other][3])
                        target_other_placement = (pos_other[0], pos_other[1], pos_other[2], wo, ho, do)
                        
                        if _box_volume(box_other) <= _box_volume(box_last):
                            continue
                            
                        last_placements_temp = bins[last_bin_id][:]
                        removed_last = False
                        for j in range(len(last_placements_temp)):
                            if last_placements_temp[j] == target_last_placement:
                                last_placements_temp.pop(j)
                                removed_last = True
                                break
                                
                        if not removed_last:
                            continue
                            
                        pos_new = _find_position_bfd(box_other, last_placements_temp, W, H, D)
                        if pos_new is None:
                            continue
                            
                        other_bin_id = result[idx_other][0]
                        other_placements_temp = bins[other_bin_id][:]
                        removed_other = False
                        for j in range(len(other_placements_temp)):
                            if other_placements_temp[j] == target_other_placement:
                                other_placements_temp.pop(j)
                                removed_other = True
                                break
                                
                        if not removed_other:
                            continue
                            
                        pos_new_last = _find_position_bfd(box_last, other_placements_temp, W, H, D)
                        if pos_new_last is None:
                            continue
                            
                        bins[last_bin_id] = last_placements_temp + [(pos_new[0], pos_new[1], pos_new[2], wo, ho, do)]
                        bins[other_bin_id] = other_placements_temp + [(pos_new_last[0], pos_new_last[1], pos_new_last[2], wl, hl, dl)]
                        
                        result[idx_last] = (other_bin_id, pos_new_last[0], pos_new_last[1], pos_new_last[2], wl, hl, dl)
                        result[idx_other] = (last_bin_id, pos_new[0], pos_new[1], pos_new[2], wo, ho, do)
                        
                        improved = True
                        break
                    if improved:
                        break
                
    # Relabel bins to be consecutive
    if bins:
        occupied_bins = sorted(set(r[0] for r in result))
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            new_bin = label_map[old_bin]
            if new_bin != old_bin:
                result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                             result[i][4], result[i][5], result[i][6])
            
    return result
