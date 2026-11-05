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
        # Check for separation along at least one axis
        if not (cx + ow <= px or px + pw <= cx or
                cy + oh <= py or py + ph <= cy or
                cz + od <= pz or pz + pd <= cz):
            return False
    return True

def _pack_greedy_optimized(boxes, container, deadline):
    """
    Pack boxes using an optimized Greedy heuristic.
    1. Compute candidate coordinates once per iteration.
    2. Use Regret (number of valid positions) as primary sorting key.
    3. Use Volume as secondary sorting key (larger first if regret is equal).
    4. Place at the Bottom-Left-Front-most valid position.
    """
    W, H, D = container
    n = len(boxes)
    placed = []
    placements = {}
    unplaced = list(range(n))
    
    # Pre-compute volumes for sorting
    volumes = [w*h*d for w, h, d in boxes]
    
    while unplaced:
        if time.time() > deadline:
            break
            
        # Generate candidate coordinates from placed boxes ONCE per iteration
        xs = {0}
        ys = {0}
        zs = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)
        
        best_box_idx = None
        best_regret = float('inf')
        best_volume = -1
        best_pos_for_box = None
        best_orient_for_box = None
        
        for idx in unplaced:
            w, h, d = boxes[idx]
            vol = volumes[idx]
            options_count = 0
            box_best_score = float('inf')
            box_best_pos = None
            box_best_orient = None
            
            orientations = _orientations(w, h, d)
            
            for ow, oh, od in orientations:
                for cx in xs:
                    if cx + ow > W:
                        continue
                    for cy in ys:
                        if cy + oh > H:
                            continue
                        for cz in zs:
                            if cz + od > D:
                                continue
                            
                            if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                                options_count += 1
                                # Bottom-Left-Front: Minimize z, then y, then x
                                # Using a weighted sum for simple comparison
                                score = cz * 10000.0 + cy * 100.0 + cx * 1.0
                                if score < box_best_score:
                                    box_best_score = score
                                    box_best_pos = (cx, cy, cz)
                                    box_best_orient = (ow, oh, od)
            
            if options_count > 0:
                # Selection Criteria:
                # 1. Lowest regret (fewest options)
                # 2. Highest volume (if regret is tied)
                if (options_count < best_regret) or \
                   (options_count == best_regret and vol > best_volume):
                    best_regret = options_count
                    best_volume = vol
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
    # Allow 115 seconds for packing, leave 5s safety margin
    deadline = start_time + 115.0
    placements = _pack_greedy_optimized(boxes, container, deadline)
    result = []
    for i in range(len(boxes)):
        if i in placements:
            x, y, z, w, h, d = placements[i]
            result.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))
    return result
