"""3D Container Loading: Fast Multi-Start Greedy with Local Improvements."""
import time
import itertools
import numpy as np
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

def _generate_candidates_fast(placed, W, H, D, max_candidates=100):
    """
    Generate candidate positions efficiently.
    Focuses on corners of existing boxes and the origin.
    """
    if not placed:
        return [(0, 0, 0)]
        
    candidates = set()
    candidates.add((0, 0, 0))
    
    # Add corners of placed boxes that are "upper" or "outer"
    for (px, py, pz, pw, ph, pd) in placed:
        # These are the 7 corners that are not (px, py, pz)
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        
        # Also add the "top" face centers? No, corners are better for packing.
        # Add positions adjacent to the box on the "floor" if possible?
        # The corner (px+pw, py, pz) covers the side.
        
    # Filter and score
    # Score: Minimize Z, then Y, then X
    scored = []
    for (cx, cy, cz) in candidates:
        if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D:
            # We want to pack tight to the bottom-left-far
            score = cz * 10000 + cy * 100 + cx
            scored.append((score, cx, cy, cz))
            
    scored.sort(key=lambda x: x[0])
    return [c[1:] for c in scored[:max_candidates]]

def _pack_greedy_fast(boxes, container, order, deadline, max_candidates=100):
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
        
        candidates = _generate_candidates_fast(placed, W, H, D, max_candidates)
        
        # Optimization: If candidates is empty (shouldn't happen with (0,0,0)), break?
        # (0,0,0) is always there.
        
        for ow, oh, od in orients:
            for (cx, cy, cz) in candidates:
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                    score = cz * 10000 + cy * 100 + cx
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

def _local_search_refine(boxes, container, placements, deadline):
    """
    Try to improve the placement by moving boxes or adding missing ones.
    This is a simple hill-climbing step.
    """
    W, H, D = container
    n = len(boxes)
    # Reconstruct placed list from placements
    placed = []
    for i in range(n):
        if i in placements:
            x, y, z, w, h, d = placements[i]
            placed.append((i, x, y, z, w, h, d))
            
    filled_vol = sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in placements)
    
    # Try to insert any unpacked boxes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if i in placements:
                continue
            w, h, d = boxes[i]
            orients = _get_orientations(w, h, d)
            
            # Generate candidates based on current placed
            candidates = _generate_candidates_fast([(x,y,z,w,h,d) for (_,x,y,z,w,h,d) in placed], W, H, D, 50)
            
            best_score = float('inf')
            best_pos = None
            best_orient = None
            
            for ow, oh, od in orients:
                for (cx, cy, cz) in candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    if _fits([(x,y,z,w,h,d) for (_,x,y,z,w,h,d) in placed], cx, cy, cz, ow, oh, od, W, H, D):
                        score = cz * 10000 + cy * 100 + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_orient = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orient
                # Add to placed
                placed.append((i, cx, cy, cz, ow, oh, od))
                placements[i] = (cx, cy, cz, ow, oh, od)
                filled_vol += w * h * d
                improved = True
                
    return placements, filled_vol

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
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
    
    # 7-12. Randomized weighted orders
    random.seed(42)
    for _ in range(6):
        # Weighted sampling: larger boxes have higher weight to be picked earlier
        weights = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
        # Add some noise to weights to create variety
        noisy_weights = [w * (1 + random.uniform(-0.2, 0.2)) for w in weights]
        order = list(range(n))
        # Sort by noisy weight descending
        order.sort(key=lambda i: noisy_weights[i], reverse=True)
        orders.append(order)
        
    best_result = None
    best_filled = 0.0
    
    for i, order in enumerate(orders):
        if time.time() > deadline:
            break
            
        # Run greedy
        placements, filled_vol = _pack_greedy_fast(boxes, container, order, deadline, max_candidates=100)
        
        # Try local search refinement if we have time
        if time.time() < deadline - 5.0: # Keep 5s margin
            placements, filled_vol = _local_search_refine(boxes, container, placements, deadline)
            
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
