"""3D Container Loading: Optimized Greedy with Bottom-Left-Far + Gap-Aware Heuristic."""
import time
import itertools
import numpy as np

def _orientations(w, h, d):
    """Return all 6 distinct orientations (permutations of dims)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

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

def _gap_score(x, y, z, w, h, d, placed, W, H, D):
    """
    Score a position by how much 'empty space' it creates around it.
    We want to maximize the minimum distance to boundaries/obstacles in the
    regions adjacent to the placed box, to create large contiguous voids.
    """
    # Distance to container boundaries
    dist_x_left = x
    dist_x_right = W - (x + w)
    dist_y_front = y
    dist_y_back = H - (y + h)
    dist_z_bottom = z
    dist_z_top = D - (z + d)
    
    # For each placed box, compute the gap between this new box and that box
    min_gap = float('inf')
    for (px, py, pz, pw, ph, pd) in placed:
        # X-axis gap
        if x + w <= px:
            gap_x = px - (x + w)
        elif px + pw <= x:
            gap_x = x - (px + pw)
        else:
            gap_x = 0  # overlap in X
        
        # Y-axis gap
        if y + h <= py:
            gap_y = py - (y + h)
        elif py + ph <= y:
            gap_y = y - (py + ph)
        else:
            gap_y = 0
        
        # Z-axis gap
        if z + d <= pz:
            gap_z = pz - (z + d)
        elif pz + pd <= z:
            gap_z = z - (pz + pd)
        else:
            gap_z = 0
        
        # The box is 'adjacent' if they touch or are close in at least one axis
        # We consider the minimum gap across all axes where they don't overlap
        # Actually, for gap creation, we care about the space between them
        # If they overlap in any axis, the gap is 0 in that direction
        # We want to maximize the minimum of the gaps in non-overlapping directions
        
        # Simpler: compute the volume of the rectangular region between them
        # But that's complex. Instead, use a heuristic:
        # The gap is the minimum of the distances in each axis where they are separated
        gaps = []
        if gap_x > 0:
            gaps.append(gap_x)
        if gap_y > 0:
            gaps.append(gap_y)
        if gap_z > 0:
            gaps.append(gap_z)
        
        if gaps:
            min_gap = min(min_gap, min(gaps))
    
    # Also consider container boundaries as "obstacles"
    boundary_gaps = [dist_x_left, dist_x_right, dist_y_front, dist_y_back, dist_z_bottom, dist_z_top]
    min_boundary_gap = min(boundary_gaps) if boundary_gaps else float('inf')
    
    # Score: we want to maximize the minimum gap (creates larger voids)
    # But we also want to pack tightly, so combine with a tightness score
    return min(min_gap, min_boundary_gap)

def _pack_greedy_gap_aware(boxes, container, order, deadline):
    """
    Pack boxes using a Greedy heuristic that considers gap creation.
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    placements = {}  # box_idx -> (x, y, z, ow, oh, od)
    filled_vol = 0.0
    
    for idx in order:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        best_score = float('inf')
        best_pos = None
        best_orient = None
        best_gap = -1
        
        # Generate candidate positions
        candidates = set([(0, 0, 0)])
        
        # Add candidates based on placed boxes
        for (px, py, pz, pw, ph, pd) in placed:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
            
        # Try all orientations
        for ow, oh, od in _orientations(w, h, d):
            for (cx, cy, cz) in candidates:
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                
                if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                    # Primary score: minimize Z, Y, X (tight packing)
                    tight_score = cz * 1000.0 + cy * 100.0 + cx * 1.0
                    
                    # Secondary score: maximize gap creation
                    gap = _gap_score(cx, cy, cz, ow, oh, od, placed, W, H, D)
                    
                    # Combined score: primarily tight, secondarily gap
                    # Use a small weight for gap to break ties
                    combined_score = tight_score - 0.001 * gap
                    
                    if combined_score < best_score:
                        best_score = combined_score
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
                        best_gap = gap
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            placements[idx] = (cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
            
    return placements, filled_vol

def _pack_greedy(boxes, container, order, deadline):
    """
    Pack boxes using a Greedy Bottom-Left-Far heuristic.
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    placements = {}  # box_idx -> (x, y, z, ow, oh, od)
    filled_vol = 0.0
    
    # Weights for scoring positions: prioritize low Z, then low Y, then low X
    w_z = 1000.0
    w_y = 100.0
    w_x = 1.0
    
    for idx in order:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        best_score = float('inf')
        best_pos = None
        best_orient = None
        
        # Generate candidate positions
        candidates = set([(0, 0, 0)])
        
        # Add candidates based on placed boxes
        for (px, py, pz, pw, ph, pd) in placed:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
            
        # Try all orientations
        for ow, oh, od in _orientations(w, h, d):
            for (cx, cy, cz) in candidates:
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                
                if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                    score = cz * w_z + cy * w_y + cx * w_x
                    
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

def _gap_fill_polish(boxes, container, placements, deadline):
    """
    Try to fit remaining boxes into gaps identified by the current placement.
    """
    W, H, D = container
    n = len(boxes)
    placed = [(placements[i][0], placements[i][1], placements[i][2], 
               placements[i][3], placements[i][4], placements[i][5]) 
              for i in range(n) if i in placements]
    
    # Try to fit each unplaced box
    for i in range(n):
        if i in placements:
            continue
        if time.time() > deadline:
            break
            
        w, h, d = boxes[i]
        best_pos = None
        best_orient = None
        best_score = float('inf')
        
        # Generate candidate positions from gaps
        candidates = set([(0, 0, 0)])
        
        # Add positions adjacent to placed boxes
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
                    # Score: prefer positions that fill gaps (lower z, y, x)
                    score = cz * 1000.0 + cy * 100.0 + cx * 1.0
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            placements[i] = (cx, cy, cz, ow, oh, od)
    
    return placements

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    start_time = time.time()
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
    
    best_result = None
    best_filled = 0.0
    
    # Try standard greedy
    for order in orders:
        if time.time() > deadline:
            break
            
        placements, filled_vol = _pack_greedy(boxes, container, order, deadline)
        
        if filled_vol > best_filled:
            best_filled = filled_vol
            best_result = placements
    
    # Try gap-aware greedy with the best order
    if best_result is not None and time.time() < deadline - 30:
        # Use the order that gave the best result
        best_order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
        placements_gap, filled_vol_gap = _pack_greedy_gap_aware(boxes, container, best_order, deadline)
        
        if filled_vol_gap > best_filled:
            best_filled = filled_vol_gap
            best_result = placements_gap
    
    # Gap-fill polish
    if best_result is not None and time.time() < deadline - 10:
        best_result = _gap_fill_polish(boxes, container, best_result, deadline)
        # Recalculate filled volume
        best_filled = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in best_result)
    
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
