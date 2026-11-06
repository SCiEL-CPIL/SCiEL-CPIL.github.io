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

def _pack_boxes(boxes, container, order_indices, time_limit):
    """
    Attempt to place boxes into the container using Extreme Point heuristic.
    Returns (result, filled_volume) where result is a list of placements.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        # Sort rotations: prefer lower height (z-dim) to stack better, then lower depth, then lower width
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    placed = []  # List of (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    # Candidate points for placement (Extreme Points)
    candidates = [(0, 0, 0)]
    
    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False

    def add_candidate(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            # Check if this point is strictly inside an existing box
            inside = False
            for ex, ey, ez, ew, eh, ed in placed:
                if x > ex and x < ex + ew and y > ey and y < ey + eh and z > ez and z < ez + ed:
                    inside = True
                    break
            if not inside:
                if (x, y, z) not in candidates:
                    candidates.append((x, y, z))

    for idx in order_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
        # Sort candidates by Z (lowest), then Y (lowest), then X (leftmost)
        valid_candidates = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            found = False
            for cx, cy, cz in valid_candidates:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                if not overlaps_any(cx, cy, cz, bw, bh, bd):
                    placed.append((cx, cy, cz, bw, bh, bd))
                    result[idx] = (0, cx, cy, cz, bw, bh, bd)
                    filled_volume += bw * bh * bd
                    
                    add_candidate(cx + bw, cy, cz)
                    add_candidate(cx, cy + bh, cz)
                    add_candidate(cx, cy, cz + bd)
                    
                    if (cx, cy, cz) in candidates:
                        candidates.remove((cx, cy, cz))
                    
                    placed_this = True
                    found = True
                    break
            
            if found:
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_volume

def _shake_improve(result, boxes, container, time_limit):
    """
    Local search: try to move/rotate placed boxes to free space for unplaced boxes.
    Returns improved (result, filled_volume).
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return result, 0
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Make a working copy
    res = [tuple(r) for r in result]
    filled_volume = sum(r[4]*r[5]*r[6] for r in res if r[0] == 0)
    
    # Identify placed and unplaced
    placed_indices = [i for i in range(n) if res[i][0] == 0]
    unplaced_indices = [i for i in range(n) if res[i][0] == -1]
    
    if not unplaced_indices:
        return res, filled_volume
    
    # Precompute rotations
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    # Build placed list from result
    def get_placed(res):
        return [(r[1], r[2], r[3], r[4], r[5], r[6]) for r in res if r[0] == 0]
    
    def check_no_overlap(placed_list, px, py, pz, pw, ph, pd, exclude_idx=-1):
        for j, (ex, ey, ez, ew, eh, ed) in enumerate(placed_list):
            if j == exclude_idx:
                continue
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return False
        return True
    
    # Try to place each unplaced box by temporarily removing some placed boxes
    max_shakes = 50
    shake_count = 0
    
    while unplaced_indices and shake_count < max_shakes:
        if time.time() > deadline:
            break
        
        placed_list = get_placed(res)
        progress = False
        
        for ui in list(unplaced_indices):
            if time.time() > deadline:
                break
            
            w, h, d = boxes[ui]
            placed_this = False
            
            # Try all rotations and candidate positions
            # Candidate positions: corners of existing boxes and container origin
            candidates = set()
            candidates.add((0, 0, 0))
            for (ex, ey, ez, ew, eh, ed) in placed_list:
                candidates.add((ex + ew, ey, ez))
                candidates.add((ex, ey + eh, ez))
                candidates.add((ex, ey, ez + ed))
                candidates.add((ex + ew, ey + eh, ez))
                candidates.add((ex, ey + eh, ez + ed))
                candidates.add((ex + ew, ey, ez + ed))
                candidates.add((ex, ey + eh, ez + ed))
                candidates.add((ex + ew, ey + eh, ez + ed))
            
            valid_candidates = [(cx, cy, cz) for cx, cy, cz in candidates 
                               if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D]
            valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
            
            for bw, bh, bd in box_rotations[ui]:
                if placed_this:
                    break
                for cx, cy, cz in valid_candidates:
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    if check_no_overlap(placed_list, cx, cy, cz, bw, bh, bd):
                        # Place the box
                        res[ui] = (0, cx, cy, cz, bw, bh, bd)
                        filled_volume += bw * bh * bd
                        unplaced_indices.remove(ui)
                        placed_this = True
                        progress = True
                        break
            
            if placed_this:
                break
        
        if not progress:
            # Try shake: remove a random placed box, try to place an unplaced one
            if placed_list and unplaced_indices:
                # Pick a small placed box to remove (less volume loss)
                placed_with_vol = [(i, res[i][4]*res[i][5]*res[i][6]) for i in placed_indices if res[i][0] == 0]
                if placed_with_vol:
                    placed_with_vol.sort(key=lambda x: x[1])
                    # Try removing the smallest few
                    for remove_idx, _ in placed_with_vol[:3]:
                        if time.time() > deadline:
                            break
                        
                        # Save old placement
                        old_placement = res[remove_idx]
                        res[remove_idx] = (-1, 0, 0, 0, boxes[remove_idx][0], boxes[remove_idx][1], boxes[remove_idx][2])
                        filled_volume -= old_placement[4]*old_placement[5]*old_placement[6]
                        placed_indices.remove(remove_idx)
                        
                        # Now try to place an unplaced box
                        placed_list = get_placed(res)
                        success = False
                        
                        for ui in list(unplaced_indices):
                            if time.time() > deadline:
                                break
                            
                            w, h, d = boxes[ui]
                            candidates = set()
                            candidates.add((0, 0, 0))
                            for (ex, ey, ez, ew, eh, ed) in placed_list:
                                candidates.add((ex + ew, ey, ez))
                                candidates.add((ex, ey + eh, ez))
                                candidates.add((ex, ey, ez + ed))
                                candidates.add((ex + ew, ey + eh, ez))
                                candidates.add((ex, ey + eh, ez + ed))
                                candidates.add((ex + ew, ey, ez + ed))
                                candidates.add((ex, ey + eh, ez + ed))
                                candidates.add((ex + ew, ey + eh, ez + ed))
                            
                            valid_candidates = [(cx, cy, cz) for cx, cy, cz in candidates 
                                               if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D]
                            valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
                            
                            for bw, bh, bd in box_rotations[ui]:
                                if success:
                                    break
                                for cx, cy, cz in valid_candidates:
                                    if cx + bw > W or cy + bh > H or cz + bd > D:
                                        continue
                                    
                                    if check_no_overlap(placed_list, cx, cy, cz, bw, bh, bd):
                                        res[ui] = (0, cx, cy, cz, bw, bh, bd)
                                        filled_volume += bw * bh * bd
                                        unplaced_indices.remove(ui)
                                        success = True
                                        break
                        
                        # Also try to re-place the removed box
                        if not success:
                            # Restore
                            res[remove_idx] = old_placement
                            filled_volume += old_placement[4]*old_placement[5]*old_placement[6]
                            placed_indices.append(remove_idx)
                        else:
                            # Try to re-place the removed box too
                            placed_list = get_placed(res)
                            re_placed = False
                            w, h, d = boxes[remove_idx]
                            candidates = set()
                            candidates.add((0, 0, 0))
                            for (ex, ey, ez, ew, eh, ed) in placed_list:
                                candidates.add((ex + ew, ey, ez))
                                candidates.add((ex, ey + eh, ez))
                                candidates.add((ex, ey, ez + ed))
                                candidates.add((ex + ew, ey + eh, ez))
                                candidates.add((ex, ey + eh, ez + ed))
                                candidates.add((ex + ew, ey, ez + ed))
                                candidates.add((ex, ey + eh, ez + ed))
                                candidates.add((ex + ew, ey + eh, ez + ed))
                            
                            valid_candidates = [(cx, cy, cz) for cx, cy, cz in candidates 
                                               if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D]
                            valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
                            
                            for bw, bh, bd in box_rotations[remove_idx]:
                                if re_placed:
                                    break
                                for cx, cy, cz in valid_candidates:
                                    if cx + bw > W or cy + bh > H or cz + bd > D:
                                        continue
                                    
                                    if check_no_overlap(placed_list, cx, cy, cz, bw, bh, bd):
                                        res[remove_idx] = (0, cx, cy, cz, bw, bh, bd)
                                        filled_volume += bw * bh * bd
                                        placed_indices.append(remove_idx)
                                        re_placed = True
                                        break
                        
                        shake_count += 1
                        if success:
                            break
        
        if not progress and shake_count >= max_shakes:
            break
    
    return res, filled_volume

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
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Strategy 1: Volume descending
    order1 = list(range(n))
    order1.sort(key=lambda i: -box_volumes[i])
    
    # Strategy 2: Max dimension descending
    order2 = list(range(n))
    order2.sort(key=lambda i: -max(boxes[i]))
    
    # Strategy 3: Min dimension descending
    order3 = list(range(n))
    order3.sort(key=lambda i: -min(boxes[i]))
    
    # Strategy 4: Sum of dimensions descending
    order4 = list(range(n))
    order4.sort(key=lambda i: -sum(boxes[i]))
    
    # Strategy 5: Volume ascending (smaller first to fill gaps)
    order5 = list(range(n))
    order5.sort(key=lambda i: box_volumes[i])
    
    # Strategy 6: Random shuffle
    random.seed(42)
    order6 = list(range(n))
    random.shuffle(order6)
    
    # Strategy 7: Sort by aspect ratio (max/min)
    order7 = list(range(n))
    order7.sort(key=lambda i: -max(boxes[i]) / max(min(boxes[i]), 1e-9))
    
    # Strategy 8: Sort by surface area
    order8 = list(range(n))
    order8.sort(key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
    
    orders = [order1, order2, order3, order4, order5, order6, order7, order8]
    
    best_result = None
    best_filled = -1
    
    # We will run _pack_boxes (Extreme Point) for each order
    # and apply shake_improve to the best few results
    
    for order in orders:
        if time.time() > deadline:
            break
            
        # Run Extreme Point
        res1, filled1 = _pack_boxes(boxes, container, order, 140.0)
        if filled1 > best_filled:
            best_filled = filled1
            best_result = res1
    
    # Apply shake improvement to the best result
    if best_result is not None and time.time() < deadline - 10.0:
        improved_result, improved_filled = _shake_improve(best_result, boxes, container, deadline - start_time - 5.0)
        if improved_filled > best_filled:
            best_filled = improved_filled
            best_result = improved_result
    
    # If we have time left, do more random shuffles
    if best_result is not None and time.time() < deadline - 15.0:
        # Try a few more random shuffles
        for _ in range(10):
            if time.time() > deadline - 10.0:
                break
            random_order = list(range(n))
            random.shuffle(random_order)
            
            res1, filled1 = _pack_boxes(boxes, container, random_order, deadline - start_time - 10.0)
            if filled1 > best_filled:
                best_filled = filled1
                best_result = res1
    
    # Final shake if we have time
    if best_result is not None and time.time() < deadline - 5.0:
        improved_result, improved_filled = _shake_improve(best_result, boxes, container, deadline - start_time - 3.0)
        if improved_filled > best_filled:
            best_filled = improved_filled
            best_result = improved_result
    
    if best_result is None:
        # Fallback: return all -1
        best_result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    
    return best_result
# EVOLVE-BLOCK-END
