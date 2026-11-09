"""3D Container Loading - Skyline + Shelf hybrid with fast candidates."""
import time
import random
import numpy as np
from itertools import permutations

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
    """Generate candidate positions from coordinate combinations of placed boxes."""
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
        candidates = set()
        for (px, py, pz, pw, ph, pd) in placed:
            x_opts = [px, px + pw]
            y_opts = [py, py + ph]
            z_opts = [pz, pz + pd]
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
        candidates = set()
        for x in xs:
            for y in ys:
                for z in zs:
                    candidates.add((x, y, z))
    
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
    """Score a position: prefer lower z, then maximize contact with existing boxes."""
    contact = 0
    for (px, py, pz, pw, ph, pd) in placed:
        if abs((x + bw) - px) < 1e-9:
            oy = min(y + bh, py + ph) - max(y, py)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if oy > 0 and oz > 0:
                contact += oy * oz
        if abs((y + bh) - py) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if ox > 0 and oz > 0:
                contact += ox * oz
        if abs((z + bd) - pz) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oy = min(y + bh, py + ph) - max(y, py)
            if ox > 0 and oy > 0:
                contact += ox * oy
        if abs(x - (px + pw)) < 1e-9:
            oy = min(y + bh, py + ph) - max(y, py)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if oy > 0 and oz > 0:
                contact += oy * oz
        if abs(y - (py + ph)) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if ox > 0 and oz > 0:
                contact += ox * oz
        if abs(z - (pz + pd)) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oy = min(y + bh, py + ph) - max(y, py)
            if ox > 0 and oy > 0:
                contact += ox * oy
    
    return (z, y, x, -contact)


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
    orders.append(('vol', vol_order))
    
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    orders.append(('maxdim', maxdim_order))
    
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    orders.append(('sa', sa_order))
    
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    orders.append(('mindim', mindim_order))
    
    compact_order = sorted(range(n), key=lambda i: max(boxes[i]) / max(1, min(boxes[i])))
    orders.append(('compact', compact_order))
    
    vol_maxdim_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] * max(boxes[i]), reverse=True)
    orders.append(('vol_maxdim', vol_maxdim_order))
    
    svr_order = sorted(range(n), key=lambda i: (2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])) / max(1, boxes[i][0]*boxes[i][1]*boxes[i][2]))
    orders.append(('svr', svr_order))
    
    # Aspect ratio based: prefer boxes with dimensions closer to container
    W, H, D = container
    aspect_order = sorted(range(n), key=lambda i: min(abs(boxes[i][0]-W)+abs(boxes[i][1]-H)+abs(boxes[i][2]-D),
                                                        abs(boxes[i][0]-W)+abs(boxes[i][1]-D)+abs(boxes[i][2]-H),
                                                        abs(boxes[i][1]-W)+abs(boxes[i][0]-H)+abs(boxes[i][2]-D),
                                                        abs(boxes[i][1]-W)+abs(boxes[i][0]-D)+abs(boxes[i][2]-H),
                                                        abs(boxes[i][2]-W)+abs(boxes[i][0]-H)+abs(boxes[i][1]-D),
                                                        abs(boxes[i][2]-W)+abs(boxes[i][1]-H)+abs(boxes[i][0]-D)))
    orders.append(('aspect', aspect_order))
    
    return orders


class ShelfPacker:
    """Shelf-based 3D packing for diversity."""
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        self.shelves = []
        self.placed_boxes = []
    
    def _can_place_in_shelf(self, shelf, x, y, bw, bh):
        if x < 0 or y < 0:
            return False
        if x + bw > self.W or y + bh > self.H:
            return False
        for (px, py, pw, ph) in shelf['boxes_2d']:
            if not (x + bw <= px or px + pw <= x):
                if not (y + bh <= py or py + ph <= y):
                    return False
        return True
    
    def find_best_shelf_position(self, bw, bh, bd):
        best_result = None
        best_score = None
        
        for shelf in self.shelves:
            z_start = shelf['z_start']
            shelf_height = shelf['height']
            
            if z_start + bd > self.D:
                continue
            if bh > shelf_height:
                continue
            
            candidates = set()
            candidates.add((0, 0))
            
            for (px, py, pw, ph) in shelf['boxes_2d']:
                if px + pw + bw <= self.W:
                    candidates.add((px + pw, py))
                    candidates.add((px + pw, py + ph))
                if px >= bw:
                    candidates.add((px - bw, py))
                if py + ph + bh <= self.H:
                    candidates.add((px, py + ph))
                if py >= bh:
                    candidates.add((px, py - bh))
            
            for (cx, cy) in candidates:
                if self._can_place_in_shelf(shelf, cx, cy, bw, bh):
                    s = (z_start + bd, cy, cx)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_result = (shelf, cx, cy, z_start)
        
        if best_result is None:
            max_z = 0
            for shelf in self.shelves:
                if shelf['z_start'] + shelf['height'] > max_z:
                    max_z = shelf['z_start'] + shelf['height']
            
            if max_z + bd <= self.D and bh <= self.H:
                new_shelf = {'z_start': max_z, 'height': bh, 'boxes_2d': []}
                self.shelves.append(new_shelf)
                new_shelf['boxes_2d'].append((0, 0, bw, bh))
                self.placed_boxes.append((0, 0, max_z, bw, bh, bd))
                best_result = (new_shelf, 0, 0, max_z)
        
        return best_result
    
    def place_in_shelf(self, shelf, x, y, bw, bh, z_start, bd):
        shelf['boxes_2d'].append((x, y, bw, bh))
        self.placed_boxes.append((x, y, z_start, bw, bh, bd))


def _greedy_shelf(order, boxes, container, deadline):
    """Greedy placement using shelf-based approach."""
    W, H, D = container
    n = len(boxes)
    packer = ShelfPacker(W, H, D)
    result = [None] * n
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        
        best_pos = None
        best_score = None
        best_rot = None
        best_shelf = None
        
        for (bw, bh, bd) in rots:
            res = packer.find_best_shelf_position(bw, bh, bd)
            if res is not None:
                shelf, cx, cy, z_start = res
                s = (z_start + bd, cy, cx)
                if best_score is None or s < best_score:
                    best_score = s
                    best_pos = (cx, cy, z_start)
                    best_rot = (bw, bh, bd)
                    best_shelf = shelf
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            if best_shelf is not None:
                packer.place_in_shelf(best_shelf, x, y, bw, bh, z, bd)
            result[idx] = (0, x, y, z, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result


def _local_search(result, boxes, container, deadline, max_iters=500):
    """Local search: try to place unplaced boxes and improve positions."""
    W, H, D = container
    n = len(boxes)
    best_result = [tuple(r) for r in result]
    best_vol = _compute_volume(best_result)
    rng = random.Random(42)
    
    def get_placed_list(result):
        placed = []
        for i in range(n):
            if result[i][0] == 0:
                x, y, z, bw, bh, bd = result[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        return placed
    
    def try_place_box(idx, placed):
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        candidates = _generate_candidates(placed, W, H, D, level=2)
        
        for (bw, bh, bd) in rots:
            for (cx, cy, cz) in candidates:
                if _fits_at_fast(cx, cy, cz, bw, bh, bd, placed, W, H, D):
                    s = (cz, cy, cx)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_pos = (cx, cy, cz)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            return True, best_pos, best_rot
        return False, None, None
    
    # Phase 1: Try to place unplaced boxes
    unplaced = [i for i in range(n) if best_result[i][0] == -1]
    attempts = 0
    while unplaced and attempts < max_iters // 3 and time.time() < deadline:
        attempts += 1
        idx = rng.choice(unplaced)
        placed = get_placed_list(best_result)
        success, pos, rot = try_place_box(idx, placed)
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
    
    # Phase 2: Move placed boxes
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
        
        success, pos, rot = try_place_box(idx, placed)
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
    for _ in range(max_iters // 3):
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
        
        success1, pos1, rot1 = try_place_box(i1, placed)
        if success1:
            placed.append(pos1 + rot1)
            success2, pos2, rot2 = try_place_box(i2, placed)
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
    deadline = time.time() + 115
    
    if n == 0:
        return []
    
    heuristic_orders = _heuristic_orders(boxes, container, n)
    
    best_result = None
    best_vol = -1
    
    # Run fast greedy with different orderings
    for name, order in heuristic_orders:
        if time.time() > deadline - 30:
            break
        result = _fast_greedy_place(order, boxes, container, deadline, candidate_level=2)
        vol = _compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Run shelf greedy with different orderings
    if time.time() < deadline - 20:
        for name, order in heuristic_orders[:4]:
            if time.time() > deadline - 15:
                break
            result = _greedy_shelf(order, boxes, container, deadline)
            vol = _compute_volume(result)
            if vol > best_vol:
                best_vol = vol
                best_result = result
    
    # Random restarts
    if time.time() < deadline - 15 and best_result is not None:
        rng_restart = random.Random(123)
        for _ in range(5):
            if time.time() > deadline - 10:
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
        max_iters = int(remaining_time * 30)
        max_iters = max(100, min(max_iters, 1500))
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters)
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
