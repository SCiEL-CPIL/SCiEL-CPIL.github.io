"""3D Container Loading: Combined heightmap + skyline approach with diverse heuristics."""
import time
import random
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container, placed):
    """Check if box fits at pos without overlap and within container."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _heightmap_fits(box_dims, pos, container, heightmap):
    """Check if box fits using heightmap for faster floor queries."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    if x + w > W or y + h > H or z + d > D:
        return False
    # Check if the box can be placed at this position given the heightmap
    for dx in range(w):
        for dy in range(h):
            if x + dx >= W or y + dy >= H:
                return False
            if heightmap[x + dx][y + dy] > z:
                return False
    return True

def _update_heightmap(heightmap, box_dims, pos):
    """Update heightmap after placing a box."""
    x, y, z = pos
    w, h, d = box_dims
    for dx in range(w):
        for dy in range(h):
            if x + dx < len(heightmap) and y + dy < len(heightmap[0]):
                new_h = z + d
                if new_h > heightmap[x + dx][y + dy]:
                    heightmap[x + dx][y + dy] = new_h

def _remove_from_heightmap(heightmap, box_dims, pos, placed):
    """Recalculate heightmap column after removing a box (simplified: rebuild)."""
    # For simplicity, we rebuild the affected columns from placed boxes
    x, y, z = pos
    w, h, d = box_dims
    for dx in range(w):
        for dy in range(h):
            if x + dx < len(heightmap) and y + dy < len(heightmap[0]):
                max_z = 0
                for (px, py, pz, pw, ph, pd) in placed:
                    if (px <= x + dx < px + pw) and (py <= y + dy < py + ph):
                        if pz + pd > max_z:
                            max_z = pz + pd
                heightmap[x + dx][y + dy] = max_z

def _candidate_positions_skyline(placed, container, mode='extreme'):
    """Generate candidate positions using skyline method."""
    W, H, D = container
    
    if not placed:
        return [(0, 0, 0)]
    
    if mode == 'extreme':
        xs = {0}
        ys = {0}
        zs = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
        xs = [x for x in xs if x < W]
        ys = [y for y in ys if y < H]
        zs = [z for z in zs if z < D]
        points = []
        for x in xs:
            for y in ys:
                for z in zs:
                    points.append((x, y, z))
        return points
    else:  # faces
        candidates = set()
        candidates.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        return [(x, y, z) for (x, y, z) in candidates if x < W and y < H and z < D]

def _candidate_positions_heightmap(heightmap, container, box_dims):
    """Generate candidate positions using heightmap - find lowest valid position."""
    W, H, D = container
    w, h, d = box_dims
    
    best_pos = None
    best_score = None
    
    for x in range(0, W - w + 1):
        for y in range(0, H - h + 1):
            # Find the lowest z where the box can be placed
            min_z = 0
            for dx in range(w):
                for dy in range(h):
                    if x + dx < W and y + dy < H:
                        if heightmap[x + dx][y + dy] > min_z:
                            min_z = heightmap[x + dx][y + dy]
            
            z = min_z
            if z + d <= D:
                score = (z, y, x)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
    
    return [best_pos] if best_pos else []

def _position_score(pos, box_dims, placed, container, strategy='compact'):
    """Score a candidate position based on the strategy."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    
    if strategy == 'compact':
        return (z, y, x)
    elif strategy == 'back':
        return (-x, -y, -z)
    elif strategy == 'corner':
        return (-(x + y + z),)
    else:
        return (z, y, x)

def _greedy_place(boxes, container, order, deadline, mode='skyline', strategy='compact', use_heightmap=False):
    """Greedy placement with given order and mode."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    heightmap = None
    if use_heightmap:
        heightmap = [[0] * H for _ in range(W)]
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        if use_heightmap:
            # Use heightmap for faster candidate generation
            best_pos = None
            best_rot = None
            best_key = None
            
            for rw, rh, rd in rots:
                candidates = _candidate_positions_heightmap(heightmap, container, (rw, rh, rd))
                for pos in candidates:
                    if _heightmap_fits((rw, rh, rd), pos, container, heightmap):
                        key = _position_score(pos, (rw, rh, rd), placed, container, strategy)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = pos
                            best_rot = (rw, rh, rd)
        else:
            candidates = _candidate_positions_skyline(placed, container, mode)
            best_pos = None
            best_rot = None
            best_key = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), pos, container, placed):
                        key = _position_score(pos, (rw, rh, rd), placed, container, strategy)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = pos
                            best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
            if use_heightmap:
                _update_heightmap(heightmap, (rw, rh, rd), (x, y, z))
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol

def _local_search(placements, boxes, container, deadline, rng, max_iters=30):
    """Try to improve by removing and re-placing boxes."""
    W, H, D = container
    n = len(boxes)
    
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    total_vol = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in loaded_indices)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        for idx in loaded_indices:
            if time.time() > deadline:
                break
            
            p = placements[idx]
            box_pos = (p[1], p[2], p[3], p[4], p[5], p[6])
            
            # Try removing this box
            try:
                placed.remove(box_pos)
            except ValueError:
                continue
            
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _candidate_positions_skyline(placed, container, 'extreme')
            
            best_pos = None
            best_rot = None
            best_key = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), pos, container, placed):
                        key = (pos[2], pos[1], pos[0])
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = pos
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                new_pos = (x, y, z, rw, rh, rd)
                
                old_score = (box_pos[2], box_pos[1], box_pos[0])
                new_score = (z, y, x)
                
                if new_score < old_score:
                    placed.append(new_pos)
                    placements[idx] = (0, x, y, z, rw, rh, rd)
                    improved = True
                else:
                    placed.append(box_pos)
            else:
                placed.append(box_pos)
    
    total_vol = 0.0
    for i in range(n):
        if placements[i][0] == 0:
            total_vol += placements[i][4] * placements[i][5] * placements[i][6]
    
    return placements, total_vol

def _shake_local_search(placements, boxes, container, deadline, rng, max_shakes=20):
    """Shake local search: randomly remove a few boxes and re-place them."""
    W, H, D = container
    n = len(boxes)
    
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    total_vol = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in loaded_indices)
    
    best_placements = [p[:] for p in placements]
    best_vol = total_vol
    
    for shake in range(max_shakes):
        if time.time() > deadline:
            break
        
        num_remove = min(rng.randint(1, 3), len(loaded_indices))
        to_remove = rng.sample(loaded_indices, num_remove)
        
        removed_boxes = []
        for idx in to_remove:
            p = placements[idx]
            box_pos = (p[1], p[2], p[3], p[4], p[5], p[6])
            try:
                placed.remove(box_pos)
            except ValueError:
                continue
            removed_boxes.append((idx, box_pos))
        
        success = True
        new_placements = [p[:] for p in placements]
        
        for idx, old_pos in removed_boxes:
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _candidate_positions_skyline(placed, container, 'extreme')
            
            best_pos = None
            best_rot = None
            best_key = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), pos, container, placed):
                        key = (pos[2], pos[1], pos[0])
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = pos
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placed.append((x, y, z, rw, rh, rd))
                new_placements[idx] = (0, x, y, z, rw, rh, rd)
            else:
                success = False
                break
        
        if success:
            new_vol = 0.0
            for i in range(n):
                if new_placements[i][0] == 0:
                    new_vol += new_placements[i][4] * new_placements[i][5] * new_placements[i][6]
            
            if new_vol > best_vol:
                best_vol = new_vol
                best_placements = [p[:] for p in new_placements]
                placed = []
                for i in range(n):
                    p = best_placements[i]
                    if p[0] == 0:
                        placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        else:
            for idx, old_pos in removed_boxes:
                try:
                    placed.remove(old_pos)
                except ValueError:
                    pass
                placed.append(old_pos)
    
    return best_placements, best_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    mins = [min(w, h, d) for w, h, d in boxes]
    maxs = [max(w, h, d) for w, h, d in boxes]
    ratios = [max(w, h, d) / max(min(w, h, d), 1e-9) for w, h, d in boxes]
    
    best_placements = None
    best_vol = 0.0
    
    sort_strategies = [
        ('vol_desc', lambda i: vols[i], True),
        ('vol_asc', lambda i: vols[i], False),
        ('min_dim_desc', lambda i: mins[i], True),
        ('min_dim_asc', lambda i: mins[i], False),
        ('max_dim_desc', lambda i: maxs[i], True),
        ('max_dim_asc', lambda i: maxs[i], False),
        ('ratio_desc', lambda i: ratios[i], True),
        ('ratio_asc', lambda i: ratios[i], False),
    ]
    
    strategies = ['compact', 'back', 'corner']
    
    # First pass: systematic exploration with both skyline and heightmap
    for sort_name, key_fn, reverse in sort_strategies[:6]:
        if time.time() > deadline:
            break
        
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)
        
        # Try with skyline
        placements, vol = _greedy_place(boxes, container, order, deadline, 'extreme', 'compact', use_heightmap=False)
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
        
        # Try with heightmap
        placements, vol = _greedy_place(boxes, container, order, deadline, 'extreme', 'compact', use_heightmap=True)
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
    
    # Local search on best
    if best_placements is not None and time.time() < deadline:
        rng = random.Random(42)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng, max_iters=20)
    
    # Random restarts
    rng = random.Random(123)
    base_order = list(range(n))
    base_order.sort(key=lambda i: vols[i], reverse=True)
    
    remaining_time = deadline - time.time()
    num_restarts = int(min(40, max(10, remaining_time / 2.0)))
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        order = base_order[:]
        num_swaps = rng.randint(2, min(15, n))
        for _ in range(num_swaps):
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                order[a], order[b] = order[b], order[a]
        
        strat = rng.choice(strategies)
        use_hm = rng.random() < 0.5
        
        placements, vol = _greedy_place(boxes, container, order, deadline, 'extreme', strat, use_heightmap=use_hm)
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
    
    # Final shake local search
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(999)
        best_placements, best_vol = _shake_local_search(best_placements, boxes, container, deadline, rng2, max_shakes=15)
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        order = list(range(n))
        order.sort(key=lambda i: vols[i], reverse=True)
        placements, vol = _greedy_place(boxes, container, order, deadline, 'extreme', 'compact', use_heightmap=False)
        if vol > 0:
            best_placements = placements
    
    return best_placements
