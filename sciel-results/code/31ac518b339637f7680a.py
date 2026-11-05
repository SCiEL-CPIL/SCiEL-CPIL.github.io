"""3D Container Loading: Enhanced Greedy with Hybrid Candidate Generation."""
import time
import itertools
import numpy as np

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

def _generate_candidates(placed, W, H, D, max_candidates=200):
    """
    Generate candidate positions for the next box.
    Uses the 'extreme' coordinates from placed boxes and container boundaries.
    Returns a list of (x, y, z) sorted by BLF score (lower is better).
    """
    x_coords = set([0, W])
    y_coords = set([0, H])
    z_coords = set([0, D])
    
    for (px, py, pz, pw, ph, pd) in placed:
        x_coords.add(px)
        x_coords.add(px + pw)
        y_coords.add(py)
        y_coords.add(py + ph)
        z_coords.add(pz)
        z_coords.add(pz + pd)
        
    # Filter coordinates within bounds
    x_coords = [x for x in x_coords if 0 <= x <= W]
    y_coords = [y for y in y_coords if 0 <= y <= H]
    z_coords = [z for z in z_coords if 0 <= z <= D]
    
    # To reduce complexity, we don't want the full cartesian product if it's too large.
    # For 60 boxes, the number of unique coords is small (< 100). 100^3 = 1M candidates is too many.
    # Strategy: 
    # 1. Always include (0,0,0).
    # 2. Include positions adjacent to existing boxes (corners).
    # 3. Include positions on the 'floor' (z=0) and 'walls' (x=0, y=0) derived from coords.
    
    candidates = set()
    candidates.add((0, 0, 0))
    
    # Add corners of placed boxes
    for (px, py, pz, pw, ph, pd) in placed:
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        
    # Add "sweep" candidates: for each x in x_coords, for each y in y_coords, 
    # find the lowest valid z. This is O(Nx * Ny * Nplaced) which is manageable.
    # But to be safe and fast, let's stick to a subset.
    # Let's prioritize candidates with low Z, then low Y, then low X.
    
    # Sort x, y, z to prioritize smaller values
    x_coords_sorted = sorted(x_coords)
    y_coords_sorted = sorted(y_coords)
    z_coords_sorted = sorted(z_coords)
    
    # Add candidates from combinations of first few x, y, z
    # This approximates the bottom-left-far strategy
    for x in x_coords_sorted[:10]:
        for y in y_coords_sorted[:10]:
            for z in z_coords_sorted[:10]:
                candidates.add((x, y, z))
                
    # Convert to list and score
    scored_candidates = []
    for (cx, cy, cz) in candidates:
        # Score: Minimize Z, then Y, then X
        # Weights: Z is most important for gravity/stacking stability, then Y, then X
        score = cz * 10000 + cy * 100 + cx
        scored_candidates.append((score, cx, cy, cz))
        
    scored_candidates.sort(key=lambda item: item[0])
    
    # Return top max_candidates
    return [c[1:] for c in scored_candidates[:max_candidates]]

def _pack_greedy(boxes, container, order, deadline, max_candidates=100):
    """
    Pack boxes using a Greedy heuristic with specific ordering.
    Returns placements dict and filled volume.
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    placements = {}  # box_idx -> (x, y, z, ow, oh, od)
    filled_vol = 0.0
    
    for idx in order:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        orients = _get_orientations(w, h, d)
        
        best_score = float('inf')
        best_pos = None
        best_orient = None
        
        # Generate candidates once per box (shared across orientations)
        # Optimization: If placed is empty, just try (0,0,0)
        if not placed:
            candidates = [(0, 0, 0)]
        else:
            candidates = _generate_candidates(placed, W, H, D, max_candidates)
        
        # Try all orientations
        for ow, oh, od in orients:
            for (cx, cy, cz) in candidates:
                # Basic bounds check for speed
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                
                # Check overlap
                if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                    # Score: Minimize Z, then Y, then X
                    # Tie-breaker: Prefer orientations with smaller height? No, let's stick to position.
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

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    start_time = time.time()
    # Safety margin: aim to finish before 140s to avoid timeout kill at 150s
    deadline = start_time + 140.0
    
    # Generate multiple sort orders
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
    
    # 6. Aspect ratio based (prefer cubes?)
    orders.append(sorted(range(n), key=lambda i: max(boxes[i])/min(boxes[i]), reverse=True))
    
    best_result = None
    best_filled = 0.0
    
    for i, order in enumerate(orders):
        if time.time() > deadline:
            break
            
        # Use slightly different max_candidates for variety? No, keep consistent.
        placements, filled_vol = _pack_greedy(boxes, container, order, deadline, max_candidates=150)
        
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
