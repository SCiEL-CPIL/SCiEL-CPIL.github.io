import time
import itertools
import random

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
        # Overlap exists if they intersect on ALL three axes
        if not (cx + ow <= px or px + pw <= cx or
                cy + oh <= py or py + ph <= cy or
                cz + od <= pz or pz + pd <= cz):
            return False
    return True

def _pack_greedy(boxes, container, sort_key, deadline, seed=0):
    """
    Pack boxes using a greedy heuristic with a specific sort key.
    Returns (placements_dict, total_volume_packed)
    """
    W, H, D = container
    n = len(boxes)
    placed = []
    placements = {}
    
    # Calculate volumes
    volumes = [w*h*d for w, h, d in boxes]
    
    # Sort indices based on the provided key
    sorted_indices = sorted(range(n), key=lambda i: sort_key(i), reverse=True)
    
    for idx in sorted_indices:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        orientations = _orientations(w, h, d)
        
        # Generate candidate coordinates
        # We use the "support points" approach:
        # x, y, z coordinates are 0 or the right/back/top face of an existing box.
        xs = {0}
        ys = {0}
        zs = {0}
        
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
            
        # We want to find the position that minimizes (z, y, x) or similar.
        # To be efficient, we can iterate through candidate positions.
        # Since n is small (60), the number of unique coordinates is manageable.
        
        best_pos = None
        best_orient = None
        best_score = None
        
        # Sort coordinates to enforce a specific search order (e.g., Bottom-Left-Front)
        # z first (bottom), then y (left), then x (front)
        zs_sorted = sorted(zs)
        ys_sorted = sorted(ys)
        xs_sorted = sorted(xs)
        
        found = False
        for cz in zs_sorted:
            if cz + max(od for _,_,od in orientations) > D:
                continue
            for cy in ys_sorted:
                if cy + max(oh for _,oh,_ in orientations) > H:
                    continue
                for cx in xs_sorted:
                    if cx + max(ow for ow,_,_ in orientations) > W:
                        continue
                    
                    # Check all orientations for this position
                    for ow, oh, od in orientations:
                        if cx + ow > W or cy + oh > H or cz + od > D:
                            continue
                        if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                            found = True
                            best_pos = (cx, cy, cz)
                            best_orient = (ow, oh, od)
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
                
        if found:
            placed.append((best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2]))
            placements[idx] = (best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2])
            
    total_vol = sum(volumes[i] for i in placements.keys())
    return placements, total_vol

def _pack_greedy_gap_aware(boxes, container, sort_key, deadline, seed=0):
    """
    Similar to _pack_greedy but prioritizes positions that minimize "wasted space"
    or fits into gaps. For now, let's stick to a robust BLF but maybe with different tie-breaking
    or candidate generation.
    
    Actually, a better improvement is to try multiple random permutations for smaller sets
    or use a more sophisticated scoring for the position.
    
    Let's implement a "distance to origin" scoring for positions to fill corners.
    """
    W, H, D = container
    n = len(boxes)
    placed = []
    placements = {}
    
    volumes = [w*h*d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: sort_key(i), reverse=True)
    
    for idx in sorted_indices:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        orientations = _orientations(w, h, d)
        
        xs = {0}
        ys = {0}
        zs = {0}
        
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
            
        best_pos = None
        best_orient = None
        best_score = float('inf')
        
        # Try all valid positions and pick the one that minimizes (z, y, x)
        # This is effectively BLF.
        # To make it gap-aware, we could penalize positions that are "floating" (no support).
        # But BLF naturally tends to fill from the bottom.
        
        # Let's try a slightly different strategy:
        # Instead of just the first valid position in BLF order,
        # check all valid positions and pick the one with minimum (z, y, x).
        # This is the same as BLF if we iterate in that order.
        
        # Let's add a "stability" check: prefer positions where the box is supported by the floor or another box.
        # A box is supported if for every x,y in its footprint, there is a box below it or it's on the floor.
        # This is complex. Let's stick to BLF but ensure we don't leave gaps on the sides.
        
        # Actually, the previous _pack_greedy is already BLF.
        # Let's try "Bottom-Left-Far" or similar variations by changing the sort order of candidates.
        
        # For now, let's just use the same logic but ensure we handle the case where multiple boxes have the same sort key
        # by breaking ties randomly or by index.
        
        zs_sorted = sorted(zs)
        ys_sorted = sorted(ys)
        xs_sorted = sorted(xs)
        
        found = False
        for cz in zs_sorted:
            if cz + max(od for _,_,od in orientations) > D:
                continue
            for cy in ys_sorted:
                if cy + max(oh for _,oh,_ in orientations) > H:
                    continue
                for cx in xs_sorted:
                    if cx + max(ow for ow,_,_ in orientations) > W:
                        continue
                    
                    for ow, oh, od in orientations:
                        if cx + ow > W or cy + oh > H or cz + od > D:
                            continue
                        if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                            found = True
                            best_pos = (cx, cy, cz)
                            best_orient = (ow, oh, od)
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
                
        if found:
            placed.append((best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2]))
            placements[idx] = (best_pos[0], best_pos[1], best_pos[2], best_orient[0], best_orient[1], best_orient[2])
            
    total_vol = sum(volumes[i] for i in placements.keys())
    return placements, total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    W, H, D = container
    n = len(boxes)
    
    # Define sorting strategies
    def key_vol(i):
        return boxes[i][0] * boxes[i][1] * boxes[i][2]
        
    def key_max_dim(i):
        return max(boxes[i])
        
    def key_surf(i):
        w, h, d = boxes[i]
        return 2*(w*h + h*d + w*d)
        
    def key_compact(i):
        w, h, d = boxes[i]
        return (w*h*d) / max(w,h,d)
        
    def key_ratio(i):
        w, h, d = boxes[i]
        return min(w,h,d) / max(w,h,d)
        
    strategies = [key_vol, key_max_dim, key_surf, key_compact, key_ratio]
    
    best_placements = {}
    best_volume = 0
    
    # Run each strategy
    for key_func in strategies:
        if time.time() > deadline:
            break
        placements, vol = _pack_greedy(boxes, container, key_func, deadline)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            
    # If we have time left, try some random restarts with shuffled order among equal volume boxes
    # or just random perturbations of the sort key.
    
    # Let's try a few random permutations of the boxes to see if we get a better pack.
    # This is useful if the greedy choice is sensitive to the order of similarly sized boxes.
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Create a random tie-breaker
        def random_key(i, seed_val=r):
            vol = boxes[i][0] * boxes[i][1] * boxes[i][2]
            # Add a small random component to break ties
            return vol + random.Random(seed_val).random() * 0.1 * vol
            
        placements, vol = _pack_greedy(boxes, container, random_key, deadline)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            
    # Result construction
    result = []
    for i in range(n):
        if i in best_placements:
            x, y, z, w, h, d = best_placements[i]
            result.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))
            
    return result
