import time
import itertools

def _orientations(w, h, d):
    """
    Generate all unique orientations of a box with dimensions (w, h, d).
    """
    dims = [w, h, d]
    seen = set()
    results = []
    for perm in itertools.permutations(dims):
        if perm not in seen:
            seen.add(perm)
            results.append(perm)
    return results

def _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
    """
    Check if a box of size (ow, oh, od) at (cx, cy, cz) fits
    without overlapping any placed box.
    """
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + ow > W or cy + oh > H or cz + od > D:
        return False
        
    for (px, py, pz, pw, ph, pd) in placed:
        if not (cx + ow <= px or px + pw <= cx or
                cy + oh <= py or py + ph <= cy or
                cz + od <= pz or pz + pd <= cz):
            return False
    return True

def _pack_greedy_regret(boxes, container, deadline):
    """
    Pack boxes using a Greedy heuristic with Regret-based ordering.
    """
    W, H, D = container
    n = len(boxes)
    placed = []
    placements = {}
    unplaced = list(range(n))
    
    while unplaced:
        if time.time() > deadline:
            break
            
        best_box_idx = None
        best_regret = float('inf')
        best_pos_for_box = None
        best_orient_for_box = None
        
        for idx in unplaced:
            w, h, d = boxes[idx]
            options_count = 0
            box_best_score = float('inf')
            box_best_pos = None
            box_best_orient = None
            
            candidates = set([(0, 0, 0)])
            for (px, py, pz, pw, ph, pd) in placed:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
                
            for ow, oh, od in _orientations(w, h, d):
                for (cx, cy, cz) in candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                        options_count += 1
                        score = cz * 1000.0 + cy * 100.0 + cx * 1.0
                        if score < box_best_score:
                            box_best_score = score
                            box_best_pos = (cx, cy, cz)
                            box_best_orient = (ow, oh, od)
            
            if options_count > 0:
                if options_count < best_regret:
                    best_regret = options_count
                    best_box_idx = idx
                    best_pos_for_box = box_best_pos
                    best_orient_for_box = box_best_orient
        
        if best_box_idx is None:
            break
            
        idx = best_box_idx
        cx, cy, cz = best_pos_for_box
        ow, oh, od = best_orient_for_box
        placed.append((cx, cy, cz, ow, oh, od))
        placements[idx] = (cx, cy, cz, ow, oh, od)
        unplaced.remove(idx)
        
    return placements

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0
    placements = _pack_greedy_regret(boxes, container, deadline)
    result = []
    for i in range(len(boxes)):
        if i in placements:
            x, y, z, w, h, d = placements[i]
            result.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))
    return result
