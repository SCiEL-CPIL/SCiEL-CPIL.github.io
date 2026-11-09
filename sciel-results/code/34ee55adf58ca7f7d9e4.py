"""3D Container Loading: Voxel-accelerated search with diverse heuristics and local improvement."""
import time
import random
import itertools
import numpy as np

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _make_voxel_grid(container, placed, resolution=2):
    """Create a 3D numpy occupancy grid. Resolution is voxels per unit."""
    W, H, D = container
    gx = max(1, int(W * resolution))
    gy = max(1, int(H * resolution))
    gz = max(1, int(D * resolution))
    grid = np.zeros((gx, gy, gz), dtype=np.int8)
    for (px, py, pz, pw, ph, pd) in placed:
        x0 = int(px * resolution)
        x1 = min(gx, int((px + pw) * resolution))
        y0 = int(py * resolution)
        y1 = min(gy, int((py + ph) * resolution))
        z0 = int(pz * resolution)
        z1 = min(gz, int((pz + pd) * resolution))
        if x1 > x0 and y1 > y0 and z1 > z0:
            grid[x0:x1, y0:y1, z0:z1] = 1
    return grid

def _fits_voxel(box_dims, pos, container, grid, resolution=2):
    """Check if box fits at pos using voxel grid. Returns True if no overlap."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    if x + w > W or y + h > H or z + d > D:
        return False
    gx = grid.shape[0]
    x0 = int(x * resolution)
    x1 = min(gx, int((x + w) * resolution))
    y0 = int(y * resolution)
    y1 = min(grid.shape[1], int((y + h) * resolution))
    z0 = int(z * resolution)
    z1 = min(grid.shape[2], int((z + d) * resolution))
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return True
    # Check if any voxel in the region is occupied
    if np.any(grid[x0:x1, y0:y1, z0:z1]):
        return False
    return True

def _fits_exact(box_dims, pos, container, placed):
    """Exact overlap check (used for validation)."""
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

def _candidate_positions(placed, container, mode='extreme'):
    """Generate candidate (x,y,z) positions from placed boxes."""
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
    elif mode == 'faces':
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
    else:  # grid
        step_x = max(1, W // 10)
        step_y = max(1, H // 10)
        step_z = max(1, D // 10)
        points = []
        for x in range(0, W, step_x):
            for y in range(0, H, step_y):
                for z in range(0, D, step_z):
                    if x < W and y < H and z < D:
                        points.append((x, y, z))
        return points

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
    elif strategy == 'fill':
        return (z, y, x)
    else:
        return (z, y, x)

def _greedy_place(boxes, container, order, deadline, mode='extreme', strategy='compact'):
    """Greedy placement with voxel-accelerated feasibility checks."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    resolution = 2
    
    # Pre-compute candidate positions incrementally
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Build voxel grid from current placed boxes
        if placed:
            grid = _make_voxel_grid(container, placed, resolution)
        else:
            grid = None
        
        candidates = _candidate_positions(placed, container, mode)
        
        best_pos = None
        best_rot = None
        best_key = None
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if grid is not None:
                    fits = _fits_voxel((rw, rh, rd), pos, container, grid, resolution)
                else:
                    fits = _fits_exact((rw, rh, rd), pos, container, placed)
                if fits:
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
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol

def _local_search(placements, boxes, container, deadline, rng, max_iters=50):
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
    resolution = 2
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
            candidates = _candidate_positions(placed, container, 'extreme')
            
            # Build voxel grid
            if placed:
                grid = _make_voxel_grid(container, placed, resolution)
            else:
                grid = None
            
            best_pos = None
            best_rot = None
            best_key = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if grid is not None:
                        fits = _fits_voxel((rw, rh, rd), pos, container, grid, resolution)
                    else:
                        fits = _fits_exact((rw, rh, rd), pos, container, placed)
                    if fits:
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
    resolution = 2
    
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
        
        # Build grid once for all re-placements
        if placed:
            grid = _make_voxel_grid(container, placed, resolution)
        else:
            grid = None
        
        for idx, old_pos in removed_boxes:
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _candidate_positions(placed, container, 'extreme')
            
            best_pos = None
            best_rot = None
            best_key = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if grid is not None:
                        fits = _fits_voxel((rw, rh, rd), pos, container, grid, resolution)
                    else:
                        fits = _fits_exact((rw, rh, rd), pos, container, placed)
                    if fits:
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
                # Update grid
                if grid is not None:
                    gx = grid.shape[0]
                    x0 = int(x * resolution)
                    x1 = min(gx, int((x + rw) * resolution))
                    y0 = int(y * resolution)
                    y1 = min(grid.shape[1], int((y + rh) * resolution))
                    z0 = int(z * resolution)
                    z1 = min(grid.shape[2], int((z + rd) * resolution))
                    if x1 > x0 and y1 > y0 and z1 > z0:
                        grid[x0:x1, y0:y1, z0:z1] = 1
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
    
    modes = ['extreme', 'faces']
    strategies = ['compact', 'back', 'corner']
    
    # First pass: systematic exploration
    for sort_name, key_fn, reverse in sort_strategies:
        if time.time() > deadline:
            break
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)
        
        for mode in modes:
            if time.time() > deadline:
                break
            for strat in strategies:
                if time.time() > deadline:
                    break
                placements, vol = _greedy_place(boxes, container, order, deadline, mode, strat)
                if vol > best_vol:
                    best_vol = vol
                    best_placements = [p[:] for p in placements]
    
    # Local search on best
    if best_placements is not None and time.time() < deadline:
        rng = random.Random(42)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng, max_iters=30)
    
    # Random restarts with perturbed orders
    rng = random.Random(123)
    base_order = list(range(n))
    base_order.sort(key=lambda i: vols[i], reverse=True)
    
    remaining_time = deadline - time.time()
    num_restarts = int(min(200, max(20, remaining_time / 2)))
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        order = base_order[:]
        num_swaps = rng.randint(2, min(10, n))
        for _ in range(num_swaps):
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                order[a], order[b] = order[b], order[a]
        
        mode = rng.choice(modes)
        strat = rng.choice(strategies)
        
        placements, vol = _greedy_place(boxes, container, order, deadline, mode, strat)
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
    
    # Final shake local search
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(999)
        best_placements, best_vol = _shake_local_search(best_placements, boxes, container, deadline, rng2, max_shakes=30)
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        order = list(range(n))
        order.sort(key=lambda i: vols[i], reverse=True)
        placements, vol = _greedy_place(boxes, container, order, deadline, 'extreme', 'compact')
        if vol > 0:
            best_placements = placements
    
    return best_placements
