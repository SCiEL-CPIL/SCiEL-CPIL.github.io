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
        # Overlap exists if they intersect on ALL three axes
        if not (cx + ow <= px or px + pw <= cx or
                cy + oh <= py or py + ph <= cy or
                cz + od <= pz or pz + pd <= cz):
            return False
    return True

def _pack_blf(boxes, container, sort_key, deadline):
    """
    Pack boxes using Bottom-Left-Front greedy heuristic with a specific sort key.
    Returns (placements_dict, total_volume_packed)
    """
    W, H, D = container
    n = len(boxes)
    placed = []
    placements = {}
    
    # Calculate volumes
    volumes = [w*h*d for w, h, d in boxes]
    
    # Sort indices based on the provided key
    # sort_key should be a function taking index i and returning a value to sort by (descending)
    sorted_indices = sorted(range(n), key=lambda i: sort_key(i), reverse=True)
    
    for idx in sorted_indices:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        orientations = _orientations(w, h, d)
        
        # Generate candidate coordinates
        # Start with 0
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
            
        # Sort candidates to enforce BLF order: Min Z, then Min Y, then Min X
        # We need to find the first valid position in this order
        # To do this efficiently, we can sort the combined list of (z, y, x)
        # But generating all combinations is expensive if sets are large.
        # However, for 60 boxes, max size of sets is ~120.
        # 120^3 = 1.7M combos. This is too many to sort and iterate in Python for each box.
        
        # Optimization:
        # Instead of all combos, we can iterate Z, then Y, then X.
        # Since we want Min Z, we sort zs.
        # For each z, we want Min Y, so sort ys.
        # For each y, we want Min X, so sort xs.
        
        zs_sorted = sorted(zs)
        ys_sorted = sorted(ys)
        xs_sorted = sorted(xs)
        
        found = False
        best_pos = None
        best_orient = None
        
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

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    W, H, D = container
    n = len(boxes)
    
    # Define sorting strategies
    # Strategy 1: Volume descending
    def key_vol(i):
        return boxes[i][0] * boxes[i][1] * boxes[i][2]
        
    # Strategy 2: Max Dimension descending
    def key_max_dim(i):
        return max(boxes[i])
        
    # Strategy 3: Surface Area descending (often correlates with size)
    def key_surf(i):
        w, h, d = boxes[i]
        return 2*(w*h + h*d + w*d)
        
    # Strategy 4: Volume / MaxDim (compactness)
    def key_compact(i):
        w, h, d = boxes[i]
        return (w*h*d) / max(w,h,d)
        
    strategies = [key_vol, key_max_dim, key_surf, key_compact]
    
    best_placements = {}
    best_volume = 0
    
    for key_func in strategies:
        if time.time() > deadline:
            break
            
        placements, vol = _pack_blf(boxes, container, key_func, deadline)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            
    # If we have time left, we could try more random permutations or local search,
    # but for now, the multiple heuristics approach is solid.
    
    result = []
    for i in range(n):
        if i in best_placements:
            x, y, z, w, h, d = best_placements[i]
            result.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))
            
    return result
