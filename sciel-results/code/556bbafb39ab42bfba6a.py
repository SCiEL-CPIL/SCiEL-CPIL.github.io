"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import permutations
from collections import defaultdict

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
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


def _face_points(placed, W, H, D):
    """Generate candidate positions from faces of placed boxes."""
    candidates = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placed:
        candidates.append((px + pw, py, pz))
        candidates.append((px, py + ph, pz))
        candidates.append((px, py, pz + pd))
    
    valid = set()
    for (cx, cy, cz) in candidates:
        if cx < 0 or cy < 0 or cz < 0:
            continue
        if cx >= W or cy >= H or cz >= D:
            continue
        inside = False
        for (px, py, pz, pw, ph, pd) in placed:
            if px <= cx < px + pw and py <= cy < py + ph and pz <= cz < pz + pd:
                inside = True
                break
        if not inside:
            valid.add((cx, cy, cz))
    return list(valid)


def _score_position(x, y, z, bw, bh, bd, W, H, D):
    """Score a position: prefer lower y (height), then lower x, then lower z."""
    return (y, z, x)


def _fast_greedy_place(order, boxes, container, deadline):
    """Fast greedy placement using face points."""
    W, H, D = container
    n = len(boxes)
    placed = []
    result = [None] * n
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        if not placed:
            candidates = [(0, 0, 0)]
        else:
            candidates = _face_points(placed, W, H, D)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (bw, bh, bd) in rots:
            for (cx, cy, cz) in candidates:
                if _fits_at(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    s = _score_position(cx, cy, cz, bw, bh, bd, W, H, D)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_pos = (cx, cy, cz)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            placed.append((x, y, z, bw, bh, bd))
            result[idx] = (0, x, y, z, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result


def _compute_volume(result, boxes, container):
    """Compute total volume of placed boxes."""
    total = 0
    for i, (bin_id, x, y, z, bw, bh, bd) in enumerate(result):
        if bin_id == 0:
            total += bw * bh * bd
    return total


def _heuristic_orders(boxes, container, n):
    """Generate multiple heuristic orderings."""
    orders = []
    
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    orders.append(vol_order)
    
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    orders.append(maxdim_order)
    
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    orders.append(sa_order)
    
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    orders.append(mindim_order)
    
    aspect_order = sorted(range(n), key=lambda i: max(boxes[i])/min(boxes[i]) if min(boxes[i])>0 else 100, reverse=False)
    orders.append(aspect_order)
    
    return orders


def _random_orders(boxes, n, count, seed):
    """Generate random orderings."""
    rng = random.Random(seed)
    orders = []
    for _ in range(count):
        order = list(range(n))
        rng.shuffle(order)
        orders.append(order)
    return orders


def _local_search(result, boxes, container, deadline, max_iters=100):
    """Local search: try to improve by moving individual boxes."""
    W, H, D = container
    n = len(boxes)
    best_result = [tuple(r) for r in result]
    best_vol = _compute_volume(best_result, boxes, container)
    rng = random.Random(42)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        placed_indices = [i for i in range(n) if best_result[i][0] == 0]
        if not placed_indices:
            break
        
        idx = rng.choice(placed_indices)
        old_placement = best_result[idx]
        
        temp_result = [tuple(r) for r in best_result]
        temp_result[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        
        placed = []
        for i in range(n):
            if i != idx and temp_result[i][0] == 0:
                x, y, z, bw, bh, bd = temp_result[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        if not placed:
            candidates = [(0, 0, 0)]
        else:
            candidates = _face_points(placed, W, H, D)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (bw, bh, bd) in rots:
            for (cx, cy, cz) in candidates:
                if _fits_at(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    s = _score_position(cx, cy, cz, bw, bh, bd, W, H, D)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_pos = (cx, cy, cz)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            new_placement = (0, x, y, z, bw, bh, bd)
            temp_result[idx] = new_placement
            new_vol = _compute_volume(temp_result, boxes, container)
            if new_vol > best_vol:
                best_result = temp_result
                best_vol = new_vol
    
    return best_result


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 110
    
    if n == 0:
        return []
    
    heuristic_orders = _heuristic_orders(boxes, container, n)
    
    best_result = None
    best_vol = -1
    
    for order in heuristic_orders:
        if time.time() > deadline:
            break
        result = _fast_greedy_place(order, boxes, container, deadline)
        vol = _compute_volume(result, boxes, container)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_result is not None:
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=150)
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
