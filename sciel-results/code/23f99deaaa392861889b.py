"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import permutations, product

def _get_rotations(w, h, d):
    """Get all unique rotations of a box, sorted by preferred orientation."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    # Sort: prefer lower z-extent (last dim) to stack better, then lower y, then lower x
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)

def _check_overlap(x, y, z, w, h, d, placed):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box."""
    for px, py, pz, pw, ph, pd in placed:
        if not (x + w <= px or px + pw <= x or
                y + h <= py or py + ph <= y or
                z + d <= pz or pz + pd <= z):
            return True
    return False

def _fits_in_container(x, y, z, w, h, d, W, H, D):
    """Check if box fits within container boundaries."""
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= W and y + h <= H and z + d <= D)

def _generate_candidates(placed, W, H, D, max_candidates=800):
    """
    Generate candidate positions from placed boxes using extreme points.
    Returns a list of candidate (x, y, z) positions.
    """
    candidates = set()
    candidates.add((0, 0, 0))
    
    for px, py, pz, pw, ph, pd in placed:
        # New extreme points from each placed box
        new_pts = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px + pw, py, pz + pd),
            (px, py + ph, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for pt in new_pts:
            if 0 <= pt[0] <= W and 0 <= pt[1] <= H and 0 <= pt[2] <= D:
                candidates.add(pt)
    
    # Convert to list and limit size
    cand_list = list(candidates)
    if len(cand_list) > max_candidates:
        # Sort by priority (closer to origin first)
        cand_list.sort(key=lambda c: c[0] + c[1] + c[2])
        cand_list = cand_list[:max_candidates]
    
    return cand_list

def _try_place(boxes, container, sort_key='volume', seed=0):
    """
    Attempt to place boxes into the container using an Extreme Point heuristic.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Sort boxes based on the chosen key
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    if sort_key == 'volume':
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    elif sort_key == 'aspect_ratio':
        def aspect(i):
            w, h, d = boxes[i]
            dims = sorted([w, h, d])
            return dims[2] / max(dims[0], 1)
        box_indices.sort(key=aspect)
    elif sort_key == 'surface_area':
        def surf(i):
            w, h, d = boxes[i]
            return 2 * (w*h + h*d + w*d)
        box_indices.sort(key=surf, reverse=True)
    elif sort_key == 'max_dim':
        def maxd(i):
            w, h, d = boxes[i]
            return max(w, h, d)
        box_indices.sort(key=maxd, reverse=True)
    elif sort_key == 'min_dim':
        def mind(i):
            w, h, d = boxes[i]
            return min(w, h, d)
        box_indices.sort(key=mind, reverse=True)
    elif sort_key == 'random':
        rng = random.Random(seed)
        rng.shuffle(box_indices)
    
    # Placements: list of (x, y, z, w, h, d) for placed boxes
    placed = []
    # Result: for each original box index, store placement or None
    result = [None] * n
    
    for idx in box_indices:
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        # Generate candidate positions
        candidates = _generate_candidates(placed, W, H, D)
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Try each rotation
        for rw, rh, rd in rots:
            # Try each candidate position
            for cx, cy, cz in candidates:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                
                # Check overlap with all placed boxes
                if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                    continue
                
                # Score: prefer positions closer to origin (lower x, y, z)
                score = cx * 1.0 + cy * 1.0 + cz * 1.0
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placed.append((bx, by, bz, bw, bh, bd))
            result[idx] = (0, bx, by, bz, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
        
        # Check time
        if time.time() > deadline:
            # Fill remaining with -1
            for i in range(n):
                if result[i] is None:
                    w_i, h_i, d_i = boxes[i]
                    result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
            break
    
    # Fill any remaining None entries
    for i in range(n):
        if result[i] is None:
            w_i, h_i, d_i = boxes[i]
            result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
    
    return result

def _try_place_diverse(boxes, container):
    """
    Try multiple sorting strategies and return the best result.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin
    
    # Define different sorting strategies
    strategies = ['volume', 'aspect_ratio', 'surface_area', 'max_dim', 'min_dim']
    
    best_result = None
    best_filled = 0
    
    for strat_idx, strategy in enumerate(strategies):
        if time.time() > deadline:
            break
        
        # Place boxes using the extreme point heuristic
        result = _try_place(boxes, container, sort_key=strategy)
        
        # Calculate filled volume
        filled = 0
        for r in result:
            if r[0] == 0:
                filled += r[4] * r[5] * r[6]
        
        if filled > best_filled:
            best_filled = filled
            best_result = result
    
    if best_result is None:
        # Fallback: all boxes left out
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    return best_result

def _try_place_with_shake(boxes, container, seed=0):
    """
    Try to improve a placement by shaking (removing and re-placing) boxes.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Get initial placement
    result = _try_place(boxes, container, sort_key='volume')
    
    # Calculate initial filled volume
    filled = sum(r[4] * r[5] * r[6] for r in result if r[0] == 0)
    
    # Build placed list from result
    placed = []
    box_to_placed_idx = {}
    for i, r in enumerate(result):
        if r[0] == 0:
            placed.append((r[1], r[2], r[3], r[4], r[5], r[6]))
            box_to_placed_idx[i] = len(placed) - 1
    
    # Precompute rotations
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Try to shake: remove a box and try to re-place it in a different position
    improved = True
    max_iterations = 50
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try to improve by moving one box at a time
        for i in range(n):
            if time.time() > deadline:
                break
            
            if result[i][0] != 0:
                continue  # Only try to move already placed boxes
            
            # Remove box i from placed
            old_pos = box_to_placed_idx.get(i)
            if old_pos is None:
                continue
            
            placed.pop(old_pos)
            # Rebuild box_to_placed_idx
            box_to_placed_idx = {}
            for j, r in enumerate(result):
                if r[0] == 0:
                    box_to_placed_idx[j] = len(box_to_placed_idx)
            
            # Try to find a better position for box i
            w_orig, h_orig, d_orig = boxes[i]
            rots = box_rotations[i]
            
            candidates = _generate_candidates(placed, W, H, D)
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for rw, rh, rd in rots:
                for cx, cy, cz in candidates:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                        continue
                    score = cx + cy + cz
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                # Check if this is an improvement (lower score than before)
                old_x, old_y, old_z = result[i][1], result[i][2], result[i][3]
                old_score = old_x + old_y + old_z
                new_score = best_pos[0] + best_pos[1] + best_pos[2]
                
                if new_score < old_score:
                    # Accept the move
                    placed.append((best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]))
                    result[i] = (0, best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2])
                    # Rebuild box_to_placed_idx
                    box_to_placed_idx = {}
                    for j, r in enumerate(result):
                        if r[0] == 0:
                            box_to_placed_idx[j] = len(box_to_placed_idx)
                    improved = True
            else:
                # Put the box back in its original position
                placed.append((result[i][1], result[i][2], result[i][3], result[i][4], result[i][5], result[i][6]))
                box_to_placed_idx[i] = len(placed) - 1
                # Rebuild box_to_placed_idx
                box_to_placed_idx = {}
                for j, r in enumerate(result):
                    if r[0] == 0:
                        box_to_placed_idx[j] = len(box_to_placed_idx)
    
    return result

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    # Try the diverse strategies
    try:
        result = _try_place_diverse(boxes, container)
    except Exception:
        result = _try_place(boxes, container, sort_key='volume')
    
    # If we have time, try to improve with shake
    if time.time() < deadline - 30.0:
        try:
            result_shake = _try_place_with_shake(boxes, container, seed=42)
            
            # Compare results
            filled_orig = sum(r[4] * r[5] * r[6] for r in result if r[0] == 0)
            filled_shake = sum(r[4] * r[5] * r[6] for r in result_shake if r[0] == 0)
            
            if filled_shake > filled_orig:
                result = result_shake
        except Exception:
            pass
    
    # Fallback to simple greedy if something went wrong
    if result is None or len(result) != n:
        placements = []
        x, y, z = 0, 0, 0
        row_h, layer_d = 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    return result
# EVOLVE-BLOCK-END
