import time
import itertools
import math

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

def _pack_greedy_optimized(boxes, container, deadline, strategy="regret"):
    """
    Pack boxes using an optimized Greedy heuristic.
    Strategies:
    - "regret": Use Regret (number of valid positions) as primary sorting key.
    - "volume": Use Volume as primary sorting key.
    - "hybrid": Use a combination of Regret and Volume.
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
        best_score = float('inf')
        best_pos_for_box = None
        best_orient_for_box = None
        
        # For each unplaced box, find the best position and compute a score
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
                # Selection Criteria based on strategy
                if strategy == "regret":
                    # 1. Lowest regret (fewest options)
                    # 2. Highest volume (if regret is tied)
                    if (options_count < best_score) or \
                       (options_count == best_score and vol > (volumes.get(best_box_idx, -1) if best_box_idx is not None else -1)):
                        best_score = options_count
                        best_box_idx = idx
                        best_pos_for_box = box_best_pos
                        best_orient_for_box = box_best_orient
                elif strategy == "volume":
                    # 1. Highest volume
                    # 2. Lowest regret (if volume is tied)
                    if (vol > volumes.get(best_box_idx, -1) if best_box_idx is not None else vol > -1) or \
                       (vol == (volumes.get(best_box_idx, -1) if best_box_idx is not None else -1) and options_count < best_score):
                        best_score = options_count
                        best_box_idx = idx
                        best_pos_for_box = box_best_pos
                        best_orient_for_box = box_best_orient
                elif strategy == "hybrid":
                    # 1. Lowest regret
                    # 2. Highest volume
                    # 3. Best position (bottom-left-front)
                    combined_score = options_count * 10000.0 - vol + box_best_score * 0.001
                    if combined_score < best_score:
                        best_score = combined_score
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

def _local_search(placements, boxes, container, deadline):
    """
    Attempt to improve the packing by moving boxes to better positions.
    This is a simple local search that tries to shift each box to a lower
    position if possible.
    """
    W, H, D = container
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to shift each box down (reduce z, then y, then x)
        for idx in list(placements.keys()):
            x, y, z, w, h, d = placements[idx]
            # Try to reduce z
            for new_z in range(z - 1, -1, -1):
                if _fits([(px, py, pz, pw, ph, pd) for i, (px, py, pz, pw, ph, pd) in placements.items() if i != idx],
                         x, y, new_z, w, h, d, W, H, D):
                    placements[idx] = (x, y, new_z, w, h, d)
                    improved = True
                    break
            if improved:
                continue
            # Try to reduce y
            for new_y in range(y - 1, -1, -1):
                if _fits([(px, py, pz, pw, ph, pd) for i, (px, py, pz, pw, ph, pd) in placements.items() if i != idx],
                         x, new_y, z, w, h, d, W, H, D):
                    placements[idx] = (x, new_y, z, w, h, d)
                    improved = True
                    break
            if improved:
                continue
            # Try to reduce x
            for new_x in range(x - 1, -1, -1):
                if _fits([(px, py, pz, pw, ph, pd) for i, (px, py, pz, pw, ph, pd) in placements.items() if i != idx],
                         new_x, y, z, w, h, d, W, H, D):
                    placements[idx] = (new_x, y, z, w, h, d)
                    improved = True
                    break
    
    return placements

def load(boxes, container):
    start_time = time.time()
    # Allow 110 seconds for packing, leave 10s safety margin
    deadline = start_time + 110.0
    
    # Try multiple strategies and keep the best result
    best_placements = {}
    best_volume = 0
    
    strategies = ["regret", "volume", "hybrid"]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        placements = _pack_greedy_optimized(boxes, container, deadline, strategy)
        
        # Apply local search to improve the packing
        placements = _local_search(placements, boxes, container, deadline)
        
        # Calculate total volume packed
        total_volume = sum(w*h*d for w, h, d in placements.values())
        
        if total_volume > best_volume:
            best_volume = total_volume
            best_placements = placements
    
    result = []
    for i in range(len(boxes)):
        if i in best_placements:
            x, y, z, w, h, d = best_placements[i]
            result.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))
    return result
