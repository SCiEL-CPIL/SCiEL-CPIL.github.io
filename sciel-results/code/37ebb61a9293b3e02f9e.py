"""3D Container Loading: model-and-optimize with fast occupancy grid and multi-start search."""
import time
import random
import itertools
import sys

def _rotations(w, h, d):
    """Return all unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
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

def _score_position(x, y, z, bw, bh, bd, W, H, D, placed, strategy='compact'):
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
    
    if strategy == 'compact':
        # Prefer lower height, then more contact, then closer to origin
        return (z + bd, y + bh, x + bw)
    elif strategy == 'low':
        return (z, y, x)
    else:
        # Prefer lower y, then more contact, then closer to origin
        return (y, -contact, x + z)

def _fast_greedy_place(order, boxes, container, deadline, candidate_level=2, score_strategy='compact'):
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
                    s = _score_position(cx, cy, cz, bw, bh, bd, W, H, D, placed, score_strategy)
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
    
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    orders.append(mindim_order)
    
    # Aspect ratio (flat boxes first)
    aspect_order = sorted(range(n), key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9))
    orders.append(aspect_order)
    
    # Random orders
    rng = random.Random(42)
    for r in range(5):
        ro = list(range(n))
        rng.shuffle(ro)
        orders.append(ro)
    
    return orders

def _local_improve(placements, boxes, container, deadline, rng, max_iters=50):
    """Local search: remove and re-insert boxes to improve packing."""
    W, H, D = container
    n = len(boxes)
    
    placed_list = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed_list.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    improved = False
    for _ in range(min(max_iters, len(loaded_indices) * 2)):
        if time.time() > deadline:
            break
        idx = rng.choice(loaded_indices)
        p = placements[idx]
        box_entry = (p[1], p[2], p[3], p[4], p[5], p[6])
        
        # Remove
        try:
            placed_list.remove(box_entry)
        except ValueError:
            placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            continue
        
        # Try to re-place
        if _place_box(placed_list, boxes, idx, container, deadline, 'extreme', 'compact'):
            new_p = placed_list[-1]
            placements[idx] = (0, new_p[0], new_p[1], new_p[2], new_p[3], new_p[4], new_p[5])
            improved = True
        else:
            # Put it back
            placed_list.append(box_entry)
    
    total_vol = sum(placements[i][4] * placements[i][5] * placements[i][6] for i in range(n) if placements[i][0] == 0)
    return placements, total_vol

def _place_box(placed_list, boxes, idx, container, deadline, position_strategy='extreme', score_strategy='compact'):
    """Try to place box idx at the best position. Returns True if placed, and updates placed_list."""
    W, H, D = container
    w, h, d = boxes[idx]
    rots = _rotations(w, h, d)
    
    if position_strategy == 'extreme':
        candidates = _candidate_positions(placed_list, container)
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
    else:
        candidates = _face_positions(placed_list, container)
    
    best_key = None
    best_pos = None
    best_rot = None
    
    for pos in candidates:
        x, y, z = pos
        for rw, rh, rd in rots:
            if _fits_check(placed_list, x, y, z, rw, rh, rd, W, H, D):
                if score_strategy == 'compact':
                    key = (z + rd, y + rh, x + rw)
                else:
                    key = (z, y, x)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = pos
                    best_rot = (rw, rh, rd)
    
    if best_pos is not None:
        placed_list.append((best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]))
        return True
    return False

def _candidate_positions(placed_list, container):
    """Generate candidate positions for placing a new box."""
    W, H, D = container
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed_list:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    xs = sorted(x for x in xs if x < W)
    ys = sorted(y for y in ys if y < H)
    zs = sorted(z for z in zs if z < D)
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _face_positions(placed_list, container):
    """Generate candidate positions from face-adjacent corners."""
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed_list:
        for dx in (0, pw):
            for dy in (0, ph):
                for dz in (0, pd):
                    x, y, z = px + dx, py + dy, pz + dz
                    if x < W and y < H and z < D:
                        candidates.add((x, y, z))
    return list(candidates)

def _fits_check(placed_list, x, y, z, bw, bh, bd, W, H, D):
    """Check if box fits at position without overlap."""
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed_list:
        if (x < px + pw and x + bw > px and
            y < py + ph and y + bh > py and
            z < pz + pd and z + bd > pz):
            return False
    return True

def _greedy_load(boxes, container, order, deadline, pos_strategy, score_strategy):
    """Greedy loading with given order and strategies."""
    W, H, D = container
    n = len(boxes)
    placed_list = []
    placements = [None] * n
    total_vol = 0.0
    
    for idx in order:
        if time.time() > deadline:
            break
        if _place_box(placed_list, boxes, idx, container, deadline, pos_strategy, score_strategy):
            # Find where we just placed it
            p = placed_list[-1]
            placements[idx] = (0, p[0], p[1], p[2], p[3], p[4], p[5])
            total_vol += p[3] * p[4] * p[5]
        else:
            w, h, d = boxes[idx]
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    
    # Generate diverse orders
    orders = _heuristic_orders(boxes, container, n)
    
    best_placements = None
    best_vol = 0.0
    
    # Try multiple candidate levels and score strategies
    candidate_levels = [1, 2, 3]
    score_strategies = ['compact', 'low', 'contact']
    
    for order in orders:
        if time.time() > deadline:
            break
        for level in candidate_levels:
            for score_s in score_strategies:
                if time.time() > deadline:
                    break
                placements, vol = _fast_greedy_place(boxes, container, order, deadline, level, score_s)
                if vol > best_vol:
                    best_vol = vol
                    best_placements = placements
    
    # Local search on best
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(123)
        best_placements, best_vol = _local_improve(best_placements, boxes, container, deadline, rng2)
    
    # Additional random restarts if time permits
    if time.time() < deadline - 20:
        rng3 = random.Random(777)
        base_order = list(range(n))
        base_order.sort(key=lambda i: vols[i], reverse=True)
        
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            order = base_order[:]
            num_swaps = rng3.randint(1, min(5, n))
            for _ in range(num_swaps):
                a, b = rng3.randrange(n), rng3.randrange(n)
                if a != b:
                    order[a], order[b] = order[b], order[a]
            
            level = rng3.choice([1, 2, 3])
            score_s = rng3.choice(['compact', 'low', 'contact'])
            placements, vol = _fast_greedy_place(boxes, container, order, deadline, level, score_s)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
        
        # Final local search
        if best_placements is not None and time.time() < deadline - 2:
            rng4 = random.Random(999)
            best_placements, best_vol = _local_improve(best_placements, boxes, container, deadline, rng4)
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
