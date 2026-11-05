"""3D Container Loading: Best-Fit greedy with multiple heuristics + gap-filling local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import defaultdict

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

def _fits_fast(placed_boxes, x, y, z, w, h, d, W, H, D):
    """Fast overlap check using early termination."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed_boxes:
        if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y or z + d <= pz or pz + pd <= z):
            return False
    return True

def _pack_best_fit(boxes, container, order, deadline):
    """
    Pack boxes using Best-Fit heuristic with optimized candidate generation.
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    placements = {}  # box_idx -> (x, y, z, ow, oh, od)
    filled_vol = 0.0
    
    # Current bounding box
    cur_max_x = 0
    cur_max_y = 0
    cur_max_z = 0
    
    for idx in order:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        best_score = float('inf')
        best_pos = None
        best_orient = None
        
        # Generate candidate coordinates from placed boxes
        x_cands = {0}
        y_cands = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            x_cands.add(px)
            x_cands.add(px + pw)
            y_cands.add(py)
            y_cands.add(py + ph)
        
        x_cands = sorted([x for x in x_cands if 0 <= x <= W])
        y_cands = sorted([y for y in y_cands if 0 <= y <= H])
        
        # For each orientation
        for ow, oh, od in _orientations(w, h, d):
            # For each candidate X, Y
            for cx in x_cands:
                if cx + ow > W:
                    continue
                for cy in y_cands:
                    if cy + oh > H:
                        continue
                    
                    # Find valid Z candidates for this (cx, cy)
                    z_cands = {0}
                    for (px, py, pz, pw, ph, pd) in placed:
                        # Check if box overlaps in XY projection
                        if cx + ow > px and px + pw > cx and cy + oh > py and py + ph > cy:
                            z_cands.add(pz + pd)
                    
                    for cz in z_cands:
                        if cz < 0 or cz + od > D:
                            continue
                        
                        # Fast fit check
                        if _fits_fast(placed, cx, cy, cz, ow, oh, od, W, H, D):
                            # Calculate BB volume score
                            new_max_x = max(cur_max_x, cx + ow)
                            new_max_y = max(cur_max_y, cy + oh)
                            new_max_z = max(cur_max_z, cz + od)
                            score = new_max_x * new_max_y * new_max_z
                            
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
            
            # Update current max
            cur_max_x = max(cur_max_x, cx + ow)
            cur_max_y = max(cur_max_y, cy + oh)
            cur_max_z = max(cur_max_z, cz + od)
            
    return placements, filled_vol

def _local_search(boxes, container, placements, deadline, max_iters=50):
    """
    Local search: try to shift each placed box to fill gaps.
    For each box, try small shifts in each direction and see if it improves the bounding box.
    """
    W, H, D = container
    n = len(boxes)
    
    # Build list of placed boxes
    placed_items = []
    for idx in range(n):
        if idx in placements:
            x, y, z, ow, oh, od = placements[idx]
            placed_items.append((idx, x, y, z, ow, oh, od))
    
    if not placed_items:
        return placements
    
    # Compute current bounding box
    max_x = max(x + ow for _, x, y, z, ow, oh, od in placed_items)
    max_y = max(y + oh for _, x, y, z, ow, oh, od in placed_items)
    max_z = max(z + od for _, x, y, z, ow, oh, od in placed_items)
    current_bb_vol = max_x * max_y * max_z
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iters and time.time() < deadline:
        improved = False
        iteration += 1
        
        for i, (idx, x, y, z, ow, oh, od) in enumerate(placed_items):
            if time.time() > deadline:
                break
            
            # Try shifting in each direction by small amounts
            # We'll try shifting to reduce the bounding box
            shifts = [
                (0, 0, 0),  # no shift
                (-ow, 0, 0), (ow, 0, 0),
                (0, -oh, 0), (0, oh, 0),
                (0, 0, -od), (0, 0, od),
                # Diagonal shifts
                (-ow, -oh, 0), (ow, -oh, 0), (-ow, oh, 0), (ow, oh, 0),
                (-ow, 0, -od), (ow, 0, -od), (-ow, 0, od), (ow, 0, od),
                (0, -oh, -od), (0, oh, -od), (0, -oh, od), (0, oh, od),
                # Corner shifts
                (-ow, -oh, -od), (ow, -oh, -od), (-ow, oh, -od), (ow, oh, -od),
                (-ow, -oh, od), (ow, -oh, od), (-ow, oh, od), (ow, oh, od),
            ]
            
            for dx, dy, dz in shifts:
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Check bounds
                if nx < 0 or ny < 0 or nz < 0:
                    continue
                if nx + ow > W or ny + oh > H or nz + od > D:
                    continue
                
                # Check if this box can be moved to (nx, ny, nz)
                # Temporarily remove from placed_items for overlap check
                new_placed = []
                for j, (jidx, jx, jy, jz, jow, joh, jod) in enumerate(placed_items):
                    if j != i:
                        new_placed.append((jx, jy, jz, jow, joh, jod))
                
                if _fits_fast(new_placed, nx, ny, nz, ow, oh, od, W, H, D):
                    # Compute new bounding box
                    new_max_x = max(nx + ow, max((jx + jow) for j, (jidx, jx, jy, jz, jow, joh, jod) in enumerate(placed_items) if j != i)) if len(placed_items) > 1 else nx + ow
                    new_max_y = max(ny + oh, max((jy + joh) for j, (jidx, jx, jy, jz, jow, joh, jod) in enumerate(placed_items) if j != i)) if len(placed_items) > 1 else ny + oh
                    new_max_z = max(nz + od, max((jz + jod) for j, (jidx, jx, jy, jz, jow, joh, jod) in enumerate(placed_items) if j != i)) if len(placed_items) > 1 else nz + od
                    new_bb_vol = new_max_x * new_max_y * new_max_z
                    
                    if new_bb_vol < current_bb_vol * 0.999:  # Require meaningful improvement
                        # Apply the move
                        placed_items[i] = (idx, nx, ny, nz, ow, oh, od)
                        placements[idx] = (nx, ny, nz, ow, oh, od)
                        current_bb_vol = new_bb_vol
                        improved = True
                        break
    
    return placements

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin
    
    # Generate multiple sort orders
    orders = []
    
    # Order 1: by volume descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    
    # Order 2: by max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # Order 3: by sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # Order 4: by aspect ratio (most elongated first)
    orders.append(sorted(range(n), key=lambda i: (max(boxes[i]) / max(min(boxes[i]), 1e-9)), reverse=True))
    
    # Order 5: by min dimension descending (flat boxes first)
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))
    
    # Order 6: by surface area descending
    orders.append(sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    best_result = None
    best_filled = 0.0
    
    for order in orders:
        if time.time() > deadline:
            break
        
        placements, filled_vol = _pack_best_fit(boxes, container, order, deadline)
        
        # Apply local search to improve
        if placements and time.time() < deadline - 10:
            placements = _local_search(boxes, container, placements, deadline, max_iters=20)
            # Recalculate filled volume after local search (should be same, but verify)
            filled_vol = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in placements)
        
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
# EVOLVE-BLOCK-END
