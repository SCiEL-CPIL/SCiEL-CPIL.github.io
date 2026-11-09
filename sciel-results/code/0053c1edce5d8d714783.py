"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import permutations
from collections import defaultdict
from heapq import heappush, heappop

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


def _rotations_sorted(w, h, d):
    """Return unique rotations sorted by max-dim ascending (more compact first)."""
    rots = _rotations(w, h, d)
    rots.sort(key=lambda r: (max(r), r[1], r[2]))
    return rots


def _fits_at_fast(x, y, z, bw, bh, bd, placed, W, H, D):
    """Faster fit check using early termination."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for i in range(len(placed)):
        px, py, pz, pw, ph, pd = placed[i]
        if px + pw <= x:
            continue
        if x + bw <= px:
            continue
        if py + ph <= y:
            continue
        if y + bh <= py:
            continue
        if pz + pd <= z:
            continue
        if z + bd <= pz:
            continue
        return False
    return True


def _generate_candidates(placed, W, H, D, level=2):
    """Generate candidate positions from coordinate combinations of placed boxes.
    level=1: face points only
    level=2: face + edge points
    level=3: all combinations (corner points too)
    """
    if not placed:
        return [(0, 0, 0)]
    
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px)
        xs.add(px + pw)
        ys.add(py)
        ys.add(py + ph)
        zs.add(pz)
        zs.add(pz + pd)
    
    xs = sorted(set(x for x in xs if 0 <= x < W))
    ys = sorted(set(y for y in ys if 0 <= y < H))
    zs = sorted(set(z for z in zs if 0 <= z < D))
    
    if level == 1:
        # Only face points: one coordinate from box face, others at 0 or aligned
        candidates = set()
        for (px, py, pz, pw, ph, pd) in placed:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
        candidates.add((0, 0, 0))
    elif level == 2:
        # Edge combinations: pick 2 coords from box faces, 1 from {0}
        candidates = set()
        for (px, py, pz, pw, ph, pd) in placed:
            x_opts = [px, px + pw]
            y_opts = [py, py + ph]
            z_opts = [pz, pz + pd]
            # Pick 2 from box, 1 from origin
            for x in x_opts:
                for y in y_opts:
                    candidates.add((x, y, 0))
                    candidates.add((x, y, pz))
                    candidates.add((x, y, pz + pd))
            for x in x_opts:
                for z in z_opts:
                    candidates.add((x, 0, z))
                    candidates.add((x, py, z))
                    candidates.add((x, py + ph, z))
            for y in y_opts:
                for z in z_opts:
                    candidates.add((0, y, z))
                    candidates.add((px, y, z))
                    candidates.add((px + pw, y, z))
        candidates.add((0, 0, 0))
    else:
        # All combinations
        candidates = set()
        for x in xs:
            for y in ys:
                for z in zs:
                    candidates.add((x, y, z))
    
    # Filter: remove points inside any placed box
    valid = []
    valid_set = set()
    for (cx, cy, cz) in candidates:
        if (cx, cy, cz) in valid_set:
            continue
        if cx >= W or cy >= H or cz >= D:
            continue
        inside = False
        for (px, py, pz, pw, ph, pd) in placed:
            if px <= cx < px + pw and py <= cy < py + ph and pz <= cz < pz + pd:
                inside = True
                break
        if not inside:
            valid_set.add((cx, cy, cz))
            valid.append((cx, cy, cz))
    
    return valid


def _score_position(x, y, z, bw, bh, bd, W, H, D, placed):
    """Score a position: prefer lower y, then maximize contact with existing boxes."""
    # Primary: lower height
    # Secondary: more contact (boxes touching on faces)
    contact = 0
    for (px, py, pz, pw, ph, pd) in placed:
        # Check face-to-face contact
        if abs((x + bw) - px) < 1e-9 and not (y + bh <= py or py + ph <= y) and not (z + bd <= pz or pz + pd <= z):
            overlap_y = min(y + bh, py + ph) - max(y, py)
            overlap_z = min(z + bd, pz + pd) - max(z, pz)
            if overlap_y > 0 and overlap_z > 0:
                contact += overlap_y * overlap_z
        if abs((y + bh) - py) < 1e-9 and not (x + bw <= px or px + pw <= x) and not (z + bd <= pz or pz + pd <= z):
            overlap_x = min(x + bw, px + pw) - max(x, px)
            overlap_z = min(z + bd, pz + pd) - max(z, pz)
            if overlap_x > 0 and overlap_z > 0:
                contact += overlap_x * overlap_z
        if abs((z + bd) - pz) < 1e-9 and not (x + bw <= px or px + pw <= x) and not (y + bh <= py or py + ph <= y):
            overlap_x = min(x + bw, px + pw) - max(x, px)
            overlap_y = min(y + bh, py + ph) - max(y, py)
            if overlap_x > 0 and overlap_y > 0:
                contact += overlap_x * overlap_y
    
    # Prefer lower y, then more contact, then closer to origin
    return (y, -contact, x + z)


def _fast_greedy_place(order, boxes, container, deadline, candidate_level=2):
    """Fast greedy placement using generated candidates."""
    W, H, D = container
    n = len(boxes)
    placed = []
    result = [None] * n
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        
        candidates = _generate_candidates(placed, W, H, D, level=candidate_level)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        for (bw, bh, bd) in rots:
            for (cx, cy, cz) in candidates:
                if _fits_at_fast(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    s = _score_position(cx, cy, cz, bw, bh, bd, W, H, D, placed)
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


def _compute_volume(result):
    """Compute total volume of placed boxes."""
    total = 0
    for r in result:
        if r[0] == 0:
            total += r[4] * r[5] * r[6]
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
    
    # Compactness: prefer more cube-like boxes first
    compact_order = sorted(range(n), key=lambda i: max(boxes[i]) / max(1, min(boxes[i])), reverse=False)
    orders.append(compact_order)
    
    # Volume-weighted by max dimension
    vol_maxdim_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] * max(boxes[i]), reverse=True)
    orders.append(vol_maxdim_order)
    
    # Surface area to volume ratio (prefer lower ratio = more compact)
    svr_order = sorted(range(n), key=lambda i: (2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])) / max(1, boxes[i][0]*boxes[i][1]*boxes[i][2]), reverse=False)
    orders.append(svr_order)
    
    return orders


def _try_place_box(idx, boxes, container, placed, deadline, candidate_level=2):
    """Try to place box idx in the container given current placed boxes.
    Returns (success, position, rotation) or (False, None, None)."""
    W, H, D = container
    w, h, d = boxes[idx]
    rots = _rotations_sorted(w, h, d)
    
    candidates = _generate_candidates(placed, W, H, D, level=candidate_level)
    
    best_pos = None
    best_score = None
    best_rot = None
    
    for (bw, bh, bd) in rots:
        for (cx, cy, cz) in candidates:
            if _fits_at_fast(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                s = _score_position(cx, cy, cz, bw, bh, bd, W, H, D, placed)
                if best_score is None or s < best_score:
                    best_score = s
                    best_pos = (cx, cy, cz)
                    best_rot = (bw, bh, bd)
    
    if best_pos is not None:
        return True, best_pos, best_rot
    return False, None, None


def _local_search(result, boxes, container, deadline, max_iters=200):
    """Enhanced local search: place unplaced, move placed, and 2-box swaps."""
    W, H, D = container
    n = len(boxes)
    best_result = [tuple(r) for r in result]
    best_vol = _compute_volume(best_result)
    rng = random.Random(42)
    
    def get_placed(result):
        placed = []
        for i in range(n):
            if result[i][0] == 0:
                x, y, z, bw, bh, bd = result[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        return placed
    
    # Phase 1: Try to place unplaced boxes
    unplaced = [i for i in range(n) if best_result[i][0] == -1]
    attempts = 0
    while unplaced and attempts < max_iters // 2 and time.time() < deadline:
        attempts += 1
        idx = rng.choice(unplaced)
        placed = get_placed(best_result)
        success, pos, rot = _try_place_box(idx, boxes, container, placed, deadline, candidate_level=2)
        if success:
            x, y, z = pos
            bw, bh, bd = rot
            temp = list(best_result)
            temp[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp]
                best_vol = new_vol
                unplaced.remove(idx)
    
    # Phase 2: Move placed boxes to find better positions
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        placed_indices = [i for i in range(n) if best_result[i][0] == 0]
        if not placed_indices:
            break
        
        idx = rng.choice(placed_indices)
        temp = list(best_result)
        temp[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        
        placed = []
        for i in range(n):
            if i != idx and temp[i][0] == 0:
                x, y, z, bw, bh, bd = temp[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        
        success, pos, rot = _try_place_box(idx, boxes, container, placed, deadline, candidate_level=2)
        if success:
            x, y, z = pos
            bw, bh, bd = rot
            temp[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp]
                best_vol = new_vol
    
    # Phase 3: 2-box swap moves
    placed_indices = [i for i in range(n) if best_result[i][0] == 0]
    for _ in range(max_iters // 2):
        if time.time() > deadline:
            break
        if len(placed_indices) < 2:
            break
        
        i1, i2 = rng.sample(placed_indices, 2)
        temp = list(best_result)
        temp[i1] = (-1, 0, 0, 0, boxes[i1][0], boxes[i1][1], boxes[i1][2])
        temp[i2] = (-1, 0, 0, 0, boxes[i2][0], boxes[i2][1], boxes[i2][2])
        
        placed = []
        for i in range(n):
            if i not in (i1, i2) and temp[i][0] == 0:
                x, y, z, bw, bh, bd = temp[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        
        # Try placing both boxes
        success1, pos1, rot1 = _try_place_box(i1, boxes, container, placed, deadline, candidate_level=2)
        if success1:
            placed.append(pos1 + rot1)
            success2, pos2, rot2 = _try_place_box(i2, boxes, container, placed, deadline, candidate_level=2)
            if success2:
                temp[i1] = (0, pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
                temp[i2] = (0, pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                new_vol = _compute_volume(temp)
                if new_vol > best_vol:
                    best_result = [tuple(r) for r in temp]
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
    
    # Run greedy with different orderings and candidate levels
    for order in heuristic_orders:
        if time.time() > deadline:
            break
        result = _fast_greedy_place(order, boxes, container, deadline, candidate_level=2)
        vol = _compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Try level 3 candidates on the best ordering if time permits
    if time.time() < deadline - 15 and best_result is not None:
        vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        result_l3 = _fast_greedy_place(vol_order, boxes, container, deadline, candidate_level=3)
        vol_l3 = _compute_volume(result_l3)
        if vol_l3 > best_vol:
            best_vol = vol_l3
            best_result = result_l3
    
    # Random restarts
    if time.time() < deadline - 20 and best_result is not None:
        rng_restart = random.Random(123)
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            random_order = list(range(n))
            rng_restart.shuffle(random_order)
            result_rand = _fast_greedy_place(random_order, boxes, container, deadline, candidate_level=2)
            vol_rand = _compute_volume(result_rand)
            if vol_rand > best_vol:
                best_vol = vol_rand
                best_result = result_rand
    
    # Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_result is not None:
        max_iters = int(remaining_time * 40)
        max_iters = max(50, min(max_iters, 800))
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters)
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
