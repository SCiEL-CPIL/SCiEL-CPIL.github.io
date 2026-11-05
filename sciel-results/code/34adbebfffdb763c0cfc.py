"""3D Container Loading: Optimized Greedy with Enhanced Local Search."""
import time
import itertools
import random

def _get_orientations(w, h, d):
    """Return unique orientations of a box (w, h, d)."""
    seen = set()
    orients = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            orients.append(perm)
    return orients

def _overlaps(x1, y1, z1, x2, y2, z2, w1, h1, d1, w2, h2, d2):
    """Check if two boxes strictly overlap."""
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    return True

def _fits(placed, x, y, z, w, h, d, W, H, D):
    """Check if box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlaps(x, y, z, px, py, pz, w, h, d, pw, ph, pd):
            return False
    return True

def _generate_candidates(placed, W, H, D):
    """
    Generate candidate positions.
    Includes corners of placed boxes, floor projections, and wall projections.
    """
    candidates = set()
    candidates.add((0, 0, 0))
    
    for (px, py, pz, pw, ph, pd) in placed:
        # 7 corners of the box
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        
        # Floor projections (z=0)
        candidates.add((px, py, 0))
        candidates.add((px + pw, py, 0))
        candidates.add((px, py + ph, 0))
        candidates.add((px + pw, py + ph, 0))
        
        # Wall projections (x=0, y=0)
        candidates.add((0, py, pz))
        candidates.add((0, py + ph, pz))
        candidates.add((0, py, pz + pd))
        candidates.add((0, py + ph, pz + pd))
        
        candidates.add((px, 0, pz))
        candidates.add((px + pw, 0, pz))
        candidates.add((px, 0, pz + pd))
        candidates.add((px + pw, 0, pz + pd))
        
        # Back wall projection (y=0) is covered by wall projections above if we consider y=0
        # Side wall projection (x=0) is covered.
        
        # Top face projections? Maybe not necessary if we have corners.
        
    # Filter valid candidates
    valid_candidates = []
    for (cx, cy, cz) in candidates:
        if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
            # Score: Minimize Z, then Y, then X (Bottom-Left-Far)
            # Using a large multiplier to ensure lexicographic order
            score = cz * 100000 + cy * 1000 + cx
            valid_candidates.append((score, cx, cy, cz))
            
    valid_candidates.sort(key=lambda x: x[0])
    return [c[1:] for c in valid_candidates]

def _pack_greedy(boxes, container, order, deadline, max_candidates=500):
    """
    Fast greedy packing.
    """
    W, H, D = container
    placed = []
    placements = {}
    filled_vol = 0.0
    
    for idx in order:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        orients = _get_orientations(w, h, d)
        
        best_score = float('inf')
        best_pos = None
        best_orient = None
        
        candidates = _generate_candidates(placed, W, H, D)
        
        for ow, oh, od in orients:
            for (cx, cy, cz) in candidates:
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                    score = cz * 100000 + cy * 1000 + cx
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            placements[idx] = (cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
            
    return placements, filled_vol

def _local_search(boxes, container, placements, deadline):
    """
    Local search to improve the placement.
    1. Try to insert any unpacked boxes.
    2. Try to move existing boxes to free up space for larger unpacked boxes.
    """
    W, H, D = container
    n = len(boxes)
    
    # Reconstruct placed list from placements
    placed_indices = list(placements.keys())
    placed = []
    for i in placed_indices:
        x, y, z, w, h, d = placements[i]
        placed.append((x, y, z, w, h, d))
        
    filled_vol = sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in placements)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to insert any unpacked boxes
        for i in range(n):
            if i in placements:
                continue
            w, h, d = boxes[idx] if False else boxes[i] # Fix: boxes[i]
            orients = _get_orientations(w, h, d)
            
            candidates = _generate_candidates(placed, W, H, D)
            
            best_score = float('inf')
            best_pos = None
            best_orient = None
            
            for ow, oh, od in orients:
                for (cx, cy, cz) in candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                        score = cz * 100000 + cy * 1000 + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_orient = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orient
                placed.append((cx, cy, cz, ow, oh, od))
                placements[i] = (cx, cy, cz, ow, oh, od)
                filled_vol += w * h * d
                improved = True
                
        # 2. Try to move existing boxes to free up space
        # This is more complex. We try to remove a box, see if a larger unpacked box fits,
        # and if so, keep the change. Otherwise, try to move the removed box to a better spot.
        if time.time() > deadline - 5:
            break
            
        for i in list(placements.keys()):
            if time.time() > deadline - 5:
                break
            x, y, z, w, h, d = placements[i]
            # Remove box i
            del placements[i]
            # Reconstruct placed list without i
            placed_without_i = []
            for j in placements:
                px, py, pz, pw, ph, pd = placements[j]
                placed_without_i.append((px, py, pz, pw, ph, pd))
                
            # Try to insert any unpacked box that is larger than the removed one
            inserted = False
            for k in range(n):
                if k in placements:
                    continue
                kw, kh, kd = boxes[k]
                if kw * kh * kd <= w * h * d:
                    continue # Only try to replace with larger or equal volume
                
                orients = _get_orientations(kw, kh, kd)
                candidates = _generate_candidates(placed_without_i, W, H, D)
                
                best_score = float('inf')
                best_pos = None
                best_orient = None
                
                for ow, oh, od in orients:
                    for (cx, cy, cz) in candidates:
                        if cx + ow > W or cy + oh > H or cz + od > D:
                            continue
                        if _fits(placed_without_i, cx, cy, cz, ow, oh, od, W, H, D):
                            score = cz * 100000 + cy * 1000 + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_orient = (ow, oh, od)
                
                if best_pos is not None:
                    cx, cy, cz = best_pos
                    ow, oh, od = best_orient
                    # Insert box k
                    placements[k] = (cx, cy, cz, ow, oh, od)
                    placed_without_i.append((cx, cy, cz, ow, oh, od))
                    filled_vol += kw * kh * kd - w * h * d
                    inserted = True
                    improved = True
                    break
            
            if not inserted:
                # Try to move box i to a better position
                orients = _get_orientations(w, h, d)
                candidates = _generate_candidates(placed_without_i, W, H, D)
                
                best_score = float('inf')
                best_pos = None
                best_orient = None
                
                for ow, oh, od in orients:
                    for (cx, cy, cz) in candidates:
                        if cx + ow > W or cy + oh > H or cz + od > D:
                            continue
                        if _fits(placed_without_i, cx, cy, cz, ow, oh, od, W, H, D):
                            score = cz * 100000 + cy * 1000 + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_orient = (ow, oh, od)
                
                if best_pos is not None:
                    cx, cy, cz = best_pos
                    ow, oh, od = best_orient
                    placements[i] = (cx, cy, cz, ow, oh, od)
                    placed_without_i.append((cx, cy, cz, ow, oh, od))
                    # Volume doesn't change, but position might improve packing for future inserts
                    # We don't mark as improved here unless we want to trigger another loop iteration
                    # But moving a box doesn't increase volume, so it's only useful if it enables future inserts.
                    # For now, we just restore it.
                else:
                    # Could not move it, restore original position
                    placements[i] = (x, y, z, w, h, d)
                    placed_without_i.append((x, y, z, w, h, d))
                    
    return placements, filled_vol

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Generate orders
    orders = []
    
    # 1. Volume descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # 4. Surface area descending
    orders.append(sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    # 5. Min dimension descending
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))
    
    # 6. Aspect ratio based
    orders.append(sorted(range(n), key=lambda i: max(boxes[i])/min(boxes[i]) if min(boxes[i])>0 else 0, reverse=True))
    
    # 7-15. Randomized weighted orders
    random.seed(42)
    for _ in range(9):
        weights = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
        noisy_weights = [w * (1 + random.uniform(-0.2, 0.2)) for w in weights]
        order = list(range(n))
        order.sort(key=lambda i: noisy_weights[i], reverse=True)
        orders.append(order)
        
    best_result = None
    best_filled = 0.0
    
    for i, order in enumerate(orders):
        if time.time() > deadline - 10: # Keep 10s for local search of last run
            break
            
        # Run greedy
        placements, filled_vol = _pack_greedy(boxes, container, order, deadline - 5, max_candidates=500)
        
        # Try local search refinement if we have time
        if time.time() < deadline - 5.0:
            placements, filled_vol = _local_search(boxes, container, placements, deadline - 2.0)
            
        if filled_vol > best_filled:
            best_filled = filled_vol
            best_result = placements
            
    # Build final result array
    result = []
    for i in range(n):
        w, h, d = boxes[i]
        if best_result is not None and i in best_result:
            x, y, z, ow, oh, od = best_result[i]
            result.append((0, x, y, z, ow, oh, od))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
            
    return result
