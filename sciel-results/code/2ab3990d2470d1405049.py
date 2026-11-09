"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples, sorted by volume."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
    """Check if a box of size (bw,bh,bd) fits at (x,y,z) without overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + bw <= px or px + pw <= x):
            if not (y + bh <= py or py + ph <= y):
                if not (z + bd <= pz or pz + pd <= z):
                    return False
    return True

def _score_position(x, y, z, bw, bh, bd, placed, W, H, D):
    """Score a position: prefer lower z, then lower y, then lower x."""
    # Higher score is better
    score = -z * 10000 - y * 100 - x
    return score

def _find_best_position(bw, bh, bd, placed, W, H, D):
    """Find the best position for a box using candidate point generation."""
    # Generate candidate x, y, z coordinates
    xs = [0]
    ys = [0]
    zs = [0]
    
    for (px, py, pz, pw, ph, pd) in placed:
        if px + pw <= W:
            xs.append(px + pw)
        if py + ph <= H:
            ys.append(py + ph)
        if pz + pd <= D:
            zs.append(pz + pd)
    
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    zs = sorted(set(zs))
    
    best_score = -float('inf')
    best_pos = None
    
    for z in zs:
        if z + bd > D:
            continue
        for y in ys:
            if y + bh > H:
                continue
            for x in xs:
                if x + bw > W:
                    continue
                if _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
                    s = _score_position(x, y, z, bw, bh, bd, placed, W, H, D)
                    if s > best_score:
                        best_score = s
                        best_pos = (x, y, z)
    
    return best_pos

def _greedy_place(boxes, container, order):
    """Place boxes in given order, trying all rotations. Returns placements list."""
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    results = [None] * len(boxes)  # index in original boxes list
    
    for idx in order:
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        best_rot = None
        best_pos = None
        best_score = -float('inf')
        
        for (rw, rh, rd) in rots:
            pos = _find_best_position(rw, rh, rd, placed, W, H, D)
            if pos is not None:
                s = _score_position(pos[0], pos[1], pos[2], rw, rh, rd, placed, W, H, D)
                if s > best_score:
                    best_score = s
                    best_rot = (rw, rh, rd)
                    best_pos = pos
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            results[idx] = (0, x, y, z, rw, rh, rd)
        else:
            results[idx] = (-1, 0, 0, 0, w, h, d)
    
    return results

def _local_search(boxes, container, initial_order, max_iters=50):
    """Try to improve placement by reordering some boxes."""
    W, H, D = container
    best_order = initial_order[:]
    best_result = _greedy_place(boxes, container, best_order)
    best_vol = sum(w*h*d for (bin_, x, y, z, w, h, d) in best_result if bin_ == 0)
    
    for iteration in range(max_iters):
        improved = False
        # Try swapping pairs
        for i in range(len(best_order) - 1):
            for j in range(i + 1, len(best_order)):
                new_order = best_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                result = _greedy_place(boxes, container, new_order)
                vol = sum(w*h*d for (bin_, x, y, z, w, h, d) in result if bin_ == 0)
                if vol > best_vol:
                    best_vol = vol
                    best_order = new_order
                    best_result = result
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_result, best_vol

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Precompute volumes
    volumes = [w*h*d for w, h, d in boxes]
    total_vol = sum(volumes)
    container_vol = W * H * D
    
    # Sort by volume descending for initial order
    order_by_vol = sorted(range(n), key=lambda i: -volumes[i])
    
    # Try multiple strategies
    best_result = None
    best_vol = 0
    
    # Strategy 1: Volume descending
    if time.time() < deadline:
        result = _greedy_place(boxes, container, order_by_vol)
        vol = sum(w*h*d for (bin_, x, y, z, w, h, d) in result if bin_ == 0)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Strategy 2: Random restarts
    random.seed(42)
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Create a random order with some bias toward larger boxes
        order = list(range(n))
        # Partially shuffle: keep large boxes early
        large_boxes = [i for i in range(n) if volumes[i] > np.median(volumes)]
        small_boxes = [i for i in range(n) if volumes[i] <= np.median(volumes)]
        
        random.shuffle(large_boxes)
        random.shuffle(small_boxes)
        
        # Mix them
        order = []
        li, si = 0, 0
        for _ in range(n):
            if li < len(large_boxes) and (si >= len(small_boxes) or random.random() < 0.7):
                order.append(large_boxes[li])
                li += 1
            else:
                order.append(small_boxes[si])
                si += 1
        
        result = _greedy_place(boxes, container, order)
        vol = sum(w*h*d for (bin_, x, y, z, w, h, d) in result if bin_ == 0)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Local search on the best result found
    if time.time() < deadline - 10 and best_result is not None:
        # Reconstruct order from best_result
        order = []
        for i in range(n):
            if best_result[i][0] == 0:
                order.append(i)
        # Add unplaced boxes at the end
        for i in range(n):
            if best_result[i][0] == -1:
                order.append(i)
        
        result, vol = _local_search(boxes, container, order, max_iters=30)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    if best_result is None:
        # Fallback: simple shelf packing
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
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
        best_result = placements
    
    return best_result

# EVOLVE-BLOCK-END
