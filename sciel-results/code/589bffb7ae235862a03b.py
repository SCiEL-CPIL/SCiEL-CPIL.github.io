"""3D Container Loading - optimized extreme point with face-intersection candidates."""
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
    # Sort: prefer lower z-extent to stack, then lower y, then lower x
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)

def _build_grid(placed, W, H, D, res=1):
    """Build a 3D occupancy grid for fast overlap checking."""
    # We'll use a dictionary-based approach: store occupied cells
    # For simplicity, use a set of (x,y,z) integer cells that are occupied
    occupied = set()
    for px, py, pz, pw, ph, pd in placed:
        for gx in range(int(px), int(px + pw) + 1):
            for gy in range(int(py), int(py + ph) + 1):
                for gz in range(int(pz), int(pz + pd) + 1):
                    occupied.add((gx, gy, gz))
    return occupied

def _check_overlap_fast(x, y, z, w, h, d, placed):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box."""
    for px, py, pz, pw, ph, pd in placed:
        if not (x + w <= px + 1e-9 or px + pw <= x + 1e-9 or
                y + h <= py + 1e-9 or py + ph <= y + 1e-9 or
                z + d <= pz + 1e-9 or pz + pd <= z + 1e-9):
            return True
    return False

def _fits_in_container(x, y, z, w, h, d, W, H, D):
    """Check if box fits within container boundaries."""
    return (x >= -1e-9 and y >= -1e-9 and z >= -1e-9 and
            x + w <= W + 1e-9 and y + h <= H + 1e-9 and z + d <= D + 1e-9)

def _generate_candidates_v2(placed, W, H, D, max_candidates=2000):
    """
    Generate candidate positions using face-intersection method.
    Candidates are points where box faces intersect, creating potential
    tight packing positions.
    Returns a list of candidate (x, y, z) positions sorted by priority.
    """
    candidates = set()
    candidates.add((0, 0, 0))
    
    # Collect all unique x, y, z coordinates from placed boxes
    x_coords = set([0, W])
    y_coords = set([0, H])
    z_coords = set([0, D])
    
    for px, py, pz, pw, ph, pd in placed:
        x_coords.add(px)
        x_coords.add(px + pw)
        y_coords.add(py)
        y_coords.add(py + ph)
        z_coords.add(pz)
        z_coords.add(pz + pd)
    
    # Generate all combinations of these coordinates
    # This creates a grid of potential positions
    for cx in x_coords:
        for cy in y_coords:
            for cz in z_coords:
                if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D:
                    candidates.add((cx, cy, cz))
    
    # Also add positions offset from each placed box's faces
    for px, py, pz, pw, ph, pd in placed:
        offsets = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px + pw, py, pz + pd),
            (px, py + ph, pz + pd),
            (px + pw, py + ph, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for pt in offsets:
            if 0 <= pt[0] <= W and 0 <= pt[1] <= H and 0 <= pt[2] <= D:
                candidates.add(pt)
    
    # Convert to list and limit size, sorted by priority
    cand_list = list(candidates)
    # Sort by z first (fill from bottom), then y, then x
    cand_list.sort(key=lambda c: (c[2], c[1], c[0]))
    
    if len(cand_list) > max_candidates:
        cand_list = cand_list[:max_candidates]
    
    return cand_list

def _try_place(boxes, container, sort_key='volume', seed=0, max_candidates=2000):
    """
    Attempt to place boxes into the container using a face-intersection heuristic.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
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
        if time.time() > deadline:
            break
            
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        # Generate candidate positions
        candidates = _generate_candidates_v2(placed, W, H, D, max_candidates)
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Try each rotation
        for rw, rh, rd in rots:
            # Try each candidate position
            for cx, cy, cz in candidates:
                if cx + rw > W + 1e-9 or cy + rh > H + 1e-9 or cz + rd > D + 1e-9:
                    continue
                
                # Check overlap with all placed boxes
                if _check_overlap_fast(cx, cy, cz, rw, rh, rd, placed):
                    continue
                
                # Score: prefer positions closer to origin, especially lower z
                score = cz * 10.0 + cy * 5.0 + cx * 1.0
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
    deadline = start_time + 108.0  # Safety margin
    
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

def _shake_improve(boxes, container, result, seed=0):
    """
    Try to improve a placement by shaking (removing and re-placing) boxes.
    Returns improved result.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return result
    
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin
    
    # Copy result
    result = [list(r) for r in result]
    
    # Precompute rotations
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Build placed list from result
    placed = []
    box_to_placed_idx = {}
    for i, r in enumerate(result):
        if r[0] == 0:
            placed.append((r[1], r[2], r[3], r[4], r[5], r[6]))
            box_to_placed_idx[i] = len(placed) - 1
    
    improved = True
    max_iterations = 30
    
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
            
            # Try to find a better position for box i
            w_orig, h_orig, d_orig = boxes[i]
            rots = box_rotations[i]
            
            candidates = _generate_candidates_v2(placed, W, H, D, 1500)
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for rw, rh, rd in rots:
                for cx, cy, cz in candidates:
                    if cx + rw > W + 1e-9 or cy + rh > H + 1e-9 or cz + rd > D + 1e-9:
                        continue
                    if _check_overlap_fast(cx, cy, cz, rw, rh, rd, placed):
                        continue
                    score = cz * 10.0 + cy * 5.0 + cx * 1.0
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                # Check if this is an improvement (lower score than before)
                old_x, old_y, old_z = result[i][1], result[i][2], result[i][3]
                old_score = old_z * 10.0 + old_y * 5.0 + old_x * 1.0
                new_score = best_pos[2] * 10.0 + best_pos[1] * 5.0 + best_pos[0] * 1.0
                
                if new_score < old_score - 1e-9:
                    # Accept the move
                    placed.append((best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]))
                    result[i] = [0, best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]]
                    improved = True
            else:
                # Put the box back in its original position
                placed.append((result[i][1], result[i][2], result[i][3], result[i][4], result[i][5], result[i][6]))
            
            # Rebuild box_to_placed_idx
            box_to_placed_idx = {}
            for j, r in enumerate(result):
                if r[0] == 0:
                    box_to_placed_idx[j] = len(box_to_placed_idx)
    
    return [tuple(r) for r in result]

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
    if time.time() < deadline - 15.0:
        try:
            result_shake = _shake_improve(boxes, container, result, seed=42)
            
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
