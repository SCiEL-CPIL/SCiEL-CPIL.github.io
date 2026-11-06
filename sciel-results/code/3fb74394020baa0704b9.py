"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box, sorted to prefer lower z then y then x."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    # Sort: prefer lower z-extent (last dim) to stack better, then lower y, then lower x
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)

def _try_place(boxes, container, sort_key='volume_desc', deadline=None, seed=None):
    """
    Attempt to place boxes into the container using a list-packing heuristic with
    extreme-point candidate positions.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes
    indexed = list(enumerate(boxes))
    if sort_key == 'volume_desc':
        indexed.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2], reverse=True)
    elif sort_key == 'max_dim_desc':
        indexed.sort(key=lambda t: max(t[1]), reverse=True)
    elif sort_key == 'min_dim_asc':
        indexed.sort(key=lambda t: min(t[1]))
    elif sort_key == 'aspect_desc':
        indexed.sort(key=lambda t: max(t[1])/max(min(t[1]),1e-9), reverse=True)
    elif sort_key == 'volume_asc':
        indexed.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2])
    elif sort_key == 'max_dim_asc':
        indexed.sort(key=lambda t: max(t[1]))
    elif sort_key == 'random':
        if seed is not None:
            random.seed(seed)
        random.shuffle(indexed)
    else:
        indexed.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2], reverse=True)
    
    placed = []  # list of (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    used_volume = 0.0
    container_vol = W * H * D
    
    for idx, (bw, bh, bd) in indexed:
        if deadline and time.time() > deadline:
            break
        rots = _get_rotations(bw, bh, bd)
        best_pos = None
        best_score = None
        
        # Generate candidate positions: origin + extreme points from placed boxes
        candidates = set()
        candidates.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            # Also try combinations of two offsets for better filling
            for (px2, py2, pz2, pw2, ph2, pd2) in placed:
                candidates.add((px + pw, py2, pz))
                candidates.add((px, py2 + ph2, pz))
                candidates.add((px, py, pz2 + pd2))
        
        # Limit candidates to avoid combinatorial explosion
        if len(candidates) > 500:
            # Keep only the most promising ones (closest to origin in some sense)
            candidates = sorted(candidates, key=lambda c: c[0]+c[1]+c[2])[:500]
        
        for (rw, rh, rd) in rots:
            for (cx, cy, cz) in candidates:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                # Check overlap with all placed boxes
                overlap = False
                for (px, py, pz, pw, ph, pd) in placed:
                    if not (cx + rw <= px or px + pw <= cx or
                            cy + rh <= py or py + ph <= cy or
                            cz + rd <= pz or pz + pd <= cz):
                        overlap = True
                        break
                if overlap:
                    continue
                # Score: prefer positions closer to origin (more compact)
                score = cx + cy + cz
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz, rw, rh, rd)
        
        if best_pos:
            cx, cy, cz, rw, rh, rd = best_pos
            placed.append((cx, cy, cz, rw, rh, rd))
            result[idx] = (0, cx, cy, cz, rw, rh, rd)
            used_volume += rw * rh * rd
    
    return result

def _shake_local_search(boxes, container, initial_result, deadline):
    """
    Take an initial placement and try to improve it by swapping or moving boxes.
    Returns improved result.
    """
    W, H, D = container
    n = len(boxes)
    
    # Parse initial result
    loaded_indices = [i for i in range(n) if initial_result[i][0] == 0]
    placed = []
    for i in loaded_indices:
        r = initial_result[i]
        placed.append((i, r[1], r[2], r[3], r[4], r[5], r[6]))
    
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    for i, x, y, z, w, h, d in placed:
        result[i] = (0, x, y, z, w, h, d)
    
    used_volume = sum(r[4]*r[5]*r[6] for r in result if r[0] == 0)
    
    # Try to improve by moving boxes to better positions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # For each loaded box, try to move it to a better (more compact) position
        for idx in range(len(placed)):
            box_idx, bx, by, bz, bw, bh, bd = placed[idx]
            
            # Remove this box temporarily
            placed_no_box = [p for j, p in enumerate(placed) if j != idx]
            
            best_pos = None
            best_score = None
            rots = _get_rotations(bw, bh, bd)
            
            # Generate candidates
            candidates = set()
            candidates.add((0, 0, 0))
            for (pi, px, py, pz, pw, ph, pd) in placed_no_box:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
            
            if len(candidates) > 300:
                candidates = sorted(candidates, key=lambda c: c[0]+c[1]+c[2])[:300]
            
            for (rw, rh, rd) in rots:
                for (cx, cy, cz) in candidates:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    overlap = False
                    for (pi, px, py, pz, pw, ph, pd) in placed_no_box:
                        if not (cx + rw <= px or px + pw <= cx or
                                cy + rh <= py or py + ph <= cy or
                                cz + rd <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    if overlap:
                        continue
                    score = cx + cy + cz
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz, rw, rh, rd)
            
            # Only accept if strictly better (more compact)
            if best_pos and best_score < bx + by + bz:
                cx, cy, cz, rw, rh, rd = best_pos
                placed[idx] = (box_idx, cx, cy, cz, rw, rh, rd)
                result[box_idx] = (0, cx, cy, cz, rw, rh, rd)
                improved = True
    
    # Try to add unloaded boxes
    unloaded = [i for i in range(n) if result[i][0] == -1]
    # Sort by volume descending
    unloaded.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    for box_idx in unloaded:
        if time.time() > deadline:
            break
        bw, bh, bd = boxes[box_idx]
        rots = _get_rotations(bw, bh, bd)
        best_pos = None
        best_score = None
        
        candidates = set()
        candidates.add((0, 0, 0))
        for (pi, px, py, pz, pw, ph, pd) in placed:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
        
        if len(candidates) > 300:
            candidates = sorted(candidates, key=lambda c: c[0]+c[1]+c[2])[:300]
        
        for (rw, rh, rd) in rots:
            for (cx, cy, cz) in candidates:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                overlap = False
                for (pi, px, py, pz, pw, ph, pd) in placed:
                    if not (cx + rw <= px or px + pw <= cx or
                            cy + rh <= py or py + ph <= cy or
                            cz + rd <= pz or pz + pd <= cz):
                        overlap = True
                        break
                if overlap:
                    continue
                score = cx + cy + cz
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz, rw, rh, rd)
        
        if best_pos:
            cx, cy, cz, rw, rh, rd = best_pos
            placed.append((box_idx, cx, cy, cz, rw, rh, rd))
            result[box_idx] = (0, cx, cy, cz, rw, rh, rd)
    
    return result

def _better_local_search(boxes, container, initial_result, deadline):
    """
    More aggressive local search: try removing and re-adding boxes in different orders,
    and try swapping positions of pairs of boxes.
    """
    W, H, D = container
    n = len(boxes)
    
    # Parse initial result
    loaded_indices = [i for i in range(n) if initial_result[i][0] == 0]
    placed = []
    for i in loaded_indices:
        r = initial_result[i]
        placed.append((i, r[1], r[2], r[3], r[4], r[5], r[6]))
    
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    for i, x, y, z, w, h, d in placed:
        result[i] = (0, x, y, z, w, h, d)
    
    best_vol = sum(r[4]*r[5]*r[6] for r in result if r[0] == 0)
    
    # Try removing the largest box and re-inserting all boxes in different orders
    if time.time() > deadline:
        return result
    
    # Strategy: remove one box, then re-pack everything
    for remove_idx in range(len(placed)):
        if time.time() > deadline:
            break
        
        box_to_remove = placed[remove_idx]
        remaining = [p for j, p in enumerate(placed) if j != remove_idx]
        
        # Re-pack remaining boxes + the removed box
        all_boxes = [p[0] for p in remaining] + [box_to_remove[0]]
        
        # Try different sort orders
        for sort_key in ['volume_desc', 'max_dim_desc']:
            if time.time() > deadline:
                break
            
            sorted_boxes = sorted(all_boxes, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
            if sort_key == 'max_dim_desc':
                sorted_boxes = sorted(all_boxes, key=lambda i: max(boxes[i]), reverse=True)
            
            new_placed = []
            for box_idx in sorted_boxes:
                bw, bh, bd = boxes[box_idx]
                rots = _get_rotations(bw, bh, bd)
                best_pos = None
                best_score = None
                
                candidates = set()
                candidates.add((0, 0, 0))
                for (pi, px, py, pz, pw, ph, pd) in new_placed:
                    candidates.add((px + pw, py, pz))
                    candidates.add((px, py + ph, pz))
                    candidates.add((px, py, pz + pd))
                
                if len(candidates) > 300:
                    candidates = sorted(candidates, key=lambda c: c[0]+c[1]+c[2])[:300]
                
                for (rw, rh, rd) in rots:
                    for (cx, cy, cz) in candidates:
                        if cx + rw > W or cy + rh > H or cz + rd > D:
                            continue
                        overlap = False
                        for (pi, px, py, pz, pw, ph, pd) in new_placed:
                            if not (cx + rw <= px or px + pw <= cx or
                                    cy + rh <= py or py + ph <= cy or
                                    cz + rd <= pz or pz + pd <= cz):
                                overlap = True
                                break
                        if overlap:
                            continue
                        score = cx + cy + cz
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, rw, rh, rd)
                
                if best_pos:
                    cx, cy, cz, rw, rh, rd = best_pos
                    new_placed.append((box_idx, cx, cy, cz, rw, rh, rd))
            
            # Check if this re-packing is better
            new_vol = sum(r[4]*r[5]*r[6] for r in new_placed)
            if new_vol > best_vol:
                best_vol = new_vol
                result = [(-1, 0, 0, 0, 0, 0, 0)] * n
                for i, x, y, z, w, h, d in new_placed:
                    result[i] = (0, x, y, z, w, h, d)
                placed = new_placed
    
    return result

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Try multiple sorting strategies and keep the best
    strategies = [
        'volume_desc',
        'max_dim_desc',
        'min_dim_asc',
        'aspect_desc',
        'volume_asc',
        'max_dim_asc',
    ]
    
    best_result = None
    best_vol = -1
    
    for strat in strategies:
        if time.time() > deadline:
            break
        result = _try_place(boxes, container, sort_key=strat, deadline=deadline)
        vol = sum(r[4]*r[5]*r[6] for r in result if r[0] == 0)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # If we have time, try a few random orderings
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        for seed in range(5):
            if time.time() > deadline:
                break
            result = _try_place(boxes, container, sort_key='random', deadline=deadline, seed=seed)
            vol = sum(r[4]*r[5]*r[6] for r in result if r[0] == 0)
            if vol > best_vol:
                best_vol = vol
                best_result = result
    
    # Apply local search shake to the best result if we have time
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_result is not None:
        shake_deadline = start_time + 140.0  # Leave 10s margin
        improved = _shake_local_search(boxes, container, best_result, shake_deadline)
        vol = sum(r[4]*r[5]*r[6] for r in improved if r[0] == 0)
        if vol > best_vol:
            best_vol = vol
            best_result = improved
    
    # Apply better local search if we still have time
    remaining_time = deadline - time.time()
    if remaining_time > 10.0 and best_result is not None:
        better_deadline = start_time + 142.0
        improved2 = _better_local_search(boxes, container, best_result, better_deadline)
        vol = sum(r[4]*r[5]*r[6] for r in improved2 if r[0] == 0)
        if vol > best_vol:
            best_vol = vol
            best_result = improved2
    
    if best_result is None:
        # Fallback: simple shelf packing
        placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    return best_result
# EVOLVE-BLOCK-END
