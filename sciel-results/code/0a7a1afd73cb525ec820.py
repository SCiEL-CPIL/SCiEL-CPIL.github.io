"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import permutations
import numpy as np

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _check_overlap(x, y, z, w, h, d, placed):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box."""
    for px, py, pz, pw, ph, pd in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return True
    return False

def _get_extreme_points(placed, W, H, D):
    """
    Generate extreme points from placed boxes.
    An extreme point is a corner of a placed box that is not inside any other box.
    """
    candidates = set()
    candidates.add((0, 0, 0))
    
    if not placed:
        return [(0, 0, 0)]
        
    for px, py, pz, pw, ph, pd in placed:
        # 8 corners of the placed box
        corners = [
            (px, py, pz),
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px + pw, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py, pz + pd),
            (px, py + ph, pz + pd),
            (px + pw, py + ph, pz + pd)
        ]
        
        for cx, cy, cz in corners:
            # Only add if within container bounds
            if cx <= W and cy <= H and cz <= D:
                # Check if this point is strictly inside any placed box
                inside = False
                for qx, qy, qz, qw, qh, qd in placed:
                    if qx < cx < qx + qw and qy < cy < qy + qh and qz < cz < qz + qd:
                        inside = True
                        break
                if not inside:
                    candidates.add((cx, cy, cz))
                    
    return list(candidates)

def _try_place_box(box_idx, boxes, box_rotations, placed, W, H, D, prefer_low_z=True):
    """
    Try to place box_idx into the space defined by 'placed'.
    Returns (success, placement_tuple) where placement_tuple is (x, y, z, w, h, d).
    Uses Best-Fit: chooses the position with the lowest Z, then Y, then X.
    """
    w, h, d = boxes[box_idx]
    rots = box_rotations[box_idx]
    
    candidates = _get_extreme_points(placed, W, H, D)
    
    best_pos = None
    best_rot = None
    best_score = None
    
    for bw, bh, bd in rots:
        for cx, cy, cz in candidates:
            if cx + bw > W or cy + bh > H or cz + bd > D:
                continue
            
            if not _check_overlap(cx, cy, cz, bw, bh, bd, placed):
                # Valid placement
                # Score: Lower Z is better. If Z same, Lower Y. If Y same, Lower X.
                score = (cz, cy, cx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (bw, bh, bd)
                    
    if best_pos is not None:
        return True, (best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2])
    return False, None

def _pack_greedy(boxes, container, order_indices, time_limit):
    """
    Greedy packing using Extreme Points and Best-Fit (Lowest Z).
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], [], 0
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        rots.sort(key=lambda r: (r[2], r[1], r[0])) # Prefer lower height
        box_rotations.append(rots)
        
    placed = []
    unplaced_indices = []
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    for idx in order_indices:
        if time.time() > deadline:
            unplaced_indices.append(idx)
            continue
            
        success, placement = _try_place_box(idx, boxes, box_rotations, placed, W, H, D)
        
        if success:
            x, y, z, bw, bh, bd = placement
            placed.append((x, y, z, bw, bh, bd))
            result[idx] = (0, x, y, z, bw, bh, bd)
            filled_volume += bw * bh * bd
        else:
            unplaced_indices.append(idx)
            result[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            
    return placed, unplaced_indices, filled_volume

def _local_search(placed, unplaced_indices, boxes, container, time_limit):
    """
    Local search to improve packing by re-inserting boxes.
    Objective: Minimize sum of Z coordinates (encourage stacking).
    """
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + time_limit
    
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
        
    # Map index to placement info
    # placed is list of tuples (x, y, z, w, h, d)
    # We need to know which box index corresponds to each placed tuple.
    # Let's reconstruct a list of (index, x, y, z, w, h, d)
    placed_info = []
    # We don't have the indices in 'placed' from _pack_greedy directly in a way that's easy to map back
    # Let's modify _pack_greedy to return indices as well, or just re-run?
    # Better: _pack_greedy should return a list of (index, x, y, z, w, h, d)
    
    # Refactor: _pack_greedy returns placed_info
    # For now, let's assume we have the indices.
    # In _pack_greedy, we know idx. Let's change return to include indices.
    
    pass # Placeholder, will implement in load

def load(boxes, container):
    """
    Main entry point: place boxes into container.
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Strategy 1: Volume descending
    order1 = list(range(n))
    order1.sort(key=lambda i: -box_volumes[i])
    
    # Strategy 2: Max dimension descending
    order2 = list(range(n))
    order2.sort(key=lambda i: -max(boxes[i]))
    
    # Strategy 3: Height (z-dim) descending - good for stacking
    order3 = list(range(n))
    order3.sort(key=lambda i: -boxes[i][2])
    
    # Strategy 4: Surface area descending
    order4 = list(range(n))
    order4.sort(key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))

    orders = [order1, order2, order3, order4]
    
    best_result = None
    best_filled = -1
    best_placed_info = None # List of (idx, x, y, z, w, h, d)
    best_unplaced = None
    
    # Precompute rotations
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
        
    # 1. Greedy Packing Phase
    for order in orders:
        if time.time() > deadline:
            break
            
        placed = [] # (x, y, z, w, h, d)
        placed_info = [] # (idx, x, y, z, w, h, d)
        unplaced_indices = []
        filled_volume = 0
        
        for idx in order:
            if time.time() > deadline:
                unplaced_indices.append(idx)
                continue
                
            success, placement = _try_place_box(idx, boxes, box_rotations, placed, W, H, D)
            
            if success:
                x, y, z, bw, bh, bd = placement
                placed.append((x, y, z, bw, bh, bd))
                placed_info.append((idx, x, y, z, bw, bh, bd))
                filled_volume += bw * bh * bd
            else:
                unplaced_indices.append(idx)
                
        if filled_volume > best_filled:
            best_filled = filled_volume
            best_placed_info = placed_info
            best_unplaced = unplaced_indices
            
    if best_placed_info is None:
        # Fallback
        return [(-1, 0, 0, 0, 0, 0, 0)] * n

    # 2. Local Search Phase
    # Objective: Minimize Sum(Z) to encourage stacking.
    # We will try to move placed boxes to lower Z positions.
    
    placed_list = list(best_placed_info) # List of (idx, x, y, z, w, h, d)
    unplaced_list = list(best_unplaced)
    
    # Helper to calculate score: Sum of Z
    def calc_score(placed_list):
        return sum(p[3] for p in placed_list) # p[3] is z
    
    current_score = calc_score(placed_list)
    
    # Local Search Loop
    search_deadline = start_time + 110.0
    random.seed(42)
    
    while time.time() < search_deadline:
        improved = False
        
        # Try to improve by moving a placed box
        if placed_list:
            # Pick a random placed box
            rand_idx_in_list = random.randint(0, len(placed_list) - 1)
            box_info = placed_list[rand_idx_in_list]
            box_idx = box_info[0]
            old_x, old_y, old_z, old_w, old_h, old_d = box_info[1], box_info[2], box_info[3], box_info[4], box_info[5], box_info[6]
            
            # Remove from placed_list
            placed_list.pop(rand_idx_in_list)
            
            # Try to find a better position for this box
            # We need to check against the remaining placed_list
            remaining_placed = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placed_list]
            
            success, new_placement = _try_place_box(box_idx, boxes, box_rotations, remaining_placed, W, H, D)
            
            if success:
                new_x, new_y, new_z, new_w, new_h, new_d = new_placement
                
                # Accept if new Z is lower, or if it allows an unplaced box to be placed
                # For simplicity, accept if new Z is strictly lower.
                if new_z < old_z:
                    # Re-add to placed_list
                    placed_list.append((box_idx, new_x, new_y, new_z, new_w, new_h, new_d))
                    improved = True
                else:
                    # Revert
                    placed_list.append(box_info)
            else:
                # Revert
                placed_list.append(box_info)
                
        # Try to place an unplaced box
        if unplaced_list:
            # Pick a random unplaced box
            rand_idx_in_unplaced = random.randint(0, len(unplaced_list) - 1)
            box_idx = unplaced_list[rand_idx_in_unplaced]
            
            placed_tuples = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placed_list]
            
            success, new_placement = _try_place_box(box_idx, boxes, box_rotations, placed_tuples, W, H, D)
            
            if success:
                # If we can place it, do it!
                unplaced_list.pop(rand_idx_in_unplaced)
                x, y, z, bw, bh, bd = new_placement
                placed_list.append((box_idx, x, y, z, bw, bh, bd))
                improved = True
                
    # Construct final result
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    for p in placed_list:
        idx, x, y, z, w, h, d = p
        result[idx] = (0, x, y, z, w, h, d)
        
    return result
# EVOLVE-BLOCK-END
