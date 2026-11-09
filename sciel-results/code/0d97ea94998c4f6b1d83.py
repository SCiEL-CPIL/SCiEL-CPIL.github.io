import time
import random
import itertools
import math

def _rotations(w, h, d):
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        points = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px, py + ph, pz + pd),
            (px + pw, py, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for (cx, cy, cz) in points:
            if cx <= W and cy <= H and cz <= D:
                candidates.add((cx, cy, cz))
    return list(candidates)

def _try_place_eap(boxes, container, order_idx, deadline, scoring_func=None, rng=None):
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0])
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _extreme_points(placed, container)
        
        best_pos = None
        best_dim = None
        best_score = None
        best_candidates = []
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                
                score = scoring_func((cx, cy, cz), (rw, rh, rd))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
                    best_candidates = [(best_pos, best_dim)]
                elif score == best_score:
                    best_candidates.append(((cx, cy, cz), (rw, rh, rd)))
        
        if best_pos is not None:
            if rng and len(best_candidates) > 1:
                best_pos, best_dim = rng.choice(best_candidates)
            
            placed.append((best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2]))
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def _shelf_place(boxes, container, order_idx, deadline, rng=None):
    """Shelf-based packing: sort by height, place in shelves."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    # Group by height (z dimension)
    by_height = sorted(order_idx, key=lambda i: boxes[i][2])
    
    z = 0
    current_shelf_h = 0
    
    for idx in by_height:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Try to place in current shelf or new shelf
        best_pos = None
        best_dim = None
        
        # Try placing at current z level
        x = 0
        y = 0
        shelf_max_h = 0
        
        for (rw, rh, rd) in rots:
            if z + rd > H:
                continue
            # Try to fit in x-y plane at this z
            cx = 0
            cy = 0
            while cx + rw <= W and cy + rh <= H:
                if _fits((rw, rh, rd), (cx, cy, z), container) and \
                   _no_overlap((rw, rh, rd), (cx, cy, z), placed):
                    best_pos = (cx, cy, z)
                    best_dim = (rw, rh, rd)
                    break
                cx += rw
                if cx + rw > W:
                    cx = 0
                    cy += rh
            if best_pos:
                break
        
        if best_pos is None:
            # Start new shelf
            z = max(placed_box_z for (_, _, _, _, _, pz, pd) in [(p[1],p[2],p[3],p[4],p[5],p[6],0) for p in placements if p[0]==0] for placed_box_z in [p[3] + p[6]]) if any(p[0]==0 for p in placements) else 0
            # Simpler: just try from z=0 upward
            found = False
            for test_z in range(0, H, 1):
                for (rw, rh, rd) in rots:
                    if test_z + rd > H:
                        continue
                    cx = 0
                    cy = 0
                    while cx + rw <= W and cy + rh <= H:
                        if _fits((rw, rh, rd), (cx, cy, test_z), container) and \
                           _no_overlap((rw, rh, rd), (cx, cy, test_z), placed):
                            best_pos = (cx, cy, test_z)
                            best_dim = (rw, rh, rd)
                            found = True
                            break
                        cx += rw
                        if cx + rw > W:
                            cx = 0
                            cy += rh
                    if found:
                        break
                if found:
                    break
            if not found:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                continue
        
        placed.append((best_pos[0], best_pos[1], best_pos[2],
                      best_dim[0], best_dim[1], best_dim[2]))
        placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2])
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def _layered_place(boxes, container, order_idx, deadline, rng=None):
    """Layered packing: sort by one dimension, place in layers."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    # Sort by largest dimension
    order = sorted(order_idx, key=lambda i: max(boxes[i]), reverse=True)
    
    z = 0
    layer_h = 0
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Orient so smallest dimension is in z (height)
        # Try all orientations, prefer smallest z extent
        best_rot = min(rots, key=lambda r: r[2])
        rw, rh, rd = best_rot
        
        # Try to place in current layer
        best_pos = None
        for (cx, cy, cz) in _extreme_points(placed, container):
            if _fits((rw, rh, rd), (cx, cy, z), container) and \
               _no_overlap((rw, rh, rd), (cx, cy, z), placed):
                best_pos = (cx, cy, z)
                break
        
        if best_pos is None:
            # Try other orientations
            for (rw2, rh2, rd2) in rots:
                if (rw2, rh2, rd2) == (rw, rh, rd):
                    continue
                for (cx, cy, cz) in _extreme_points(placed, container):
                    if _fits((rw2, rh2, rd2), (cx, cy, z), container) and \
                       _no_overlap((rw2, rh2, rd2), (cx, cy, z), placed):
                        best_pos = (cx, cy, z)
                        rw, rh, rd = rw2, rh2, rd2
                        break
                if best_pos:
                    break
        
        if best_pos is None:
            # Start new layer
            z = max((p[3] + p[6] for p in placements if p[0] == 0), default=0)
            for (rw3, rh3, rd3) in rots:
                if _fits((rw3, rh3, rd3), (0, 0, z), container) and \
                   _no_overlap((rw3, rh3, rd3), (0, 0, z), placed):
                    best_pos = (0, 0, z)
                    rw, rh, rd = rw3, rh3, rd3
                    break
        
        if best_pos is not None:
            placed.append((best_pos[0], best_pos[1], best_pos[2],
                          rw, rh, rd))
            placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                              rw, rh, rd)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def _simulated_annealing_refine(boxes, container, initial_placements, deadline, rng):
    """Refine using simulated annealing: relocate boxes, swap positions."""
    W, H, D = container
    n = len(boxes)
    
    current_placements = [tuple(p) for p in initial_placements]
    current_volume = sum(p[4]*p[5]*p[6] for p in current_placements if p[0]==0)
    best_placements = [tuple(p) for p in current_placements]
    best_volume = current_volume
    
    T = 100.0
    cooling = 0.98
    start_time = time.time()
    
    iteration = 0
    while time.time() < deadline and T > 0.1:
        iteration += 1
        
        # Choose a random placed box to relocate
        placed_idx = [i for i, p in enumerate(current_placements) if p[0] == 0]
        if not placed_idx:
            break
        
        idx = rng.choice(placed_idx)
        old_pl = current_placements[idx]
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Remove box
        current_placements[idx] = (-1, 0, 0, 0, w, h, d)
        current_volume -= old_pl[4]*old_pl[5]*old_pl[6]
        
        # Try to re-insert at a random extreme point
        placed_list = [tuple(p[1:7]) for p in current_placements if p[0] == 0]
        candidates = _extreme_points(placed_list, container)
        
        if not candidates:
            candidates = [(0, 0, 0)]
        
        # Pick a random candidate
        cx, cy, cz = rng.choice(candidates)
        
        # Try a random rotation
        rw, rh, rd = rng.choice(rots)
        
        success = False
        if _fits((rw, rh, rd), (cx, cy, cz), container) and \
           _no_overlap((rw, rh, rd), (cx, cy, cz), placed_list):
            new_pl = (0, cx, cy, cz, rw, rh, rd)
            new_volume = current_volume + rw*rh*rd
            
            # Simulated annealing acceptance
            delta = new_volume - current_volume
            if delta >= 0 or rng.random() < math.exp(delta / max(T, 0.01)):
                current_placements[idx] = new_pl
                current_volume = new_volume
                success = True
                
                if current_volume > best_volume:
                    best_volume = current_volume
                    best_placements = [tuple(p) for p in current_placements]
        
        if not success:
            # Revert
            current_placements[idx] = old_pl
            current_volume += old_pl[4]*old_pl[5]*old_pl[6]
        
        T *= cooling
    
    return best_placements, best_volume

def _pairwise_swap(boxes, container, placements, deadline, rng, max_swaps=50):
    """Try swapping positions of two placed boxes."""
    W, H, D = container
    n = len(boxes)
    
    current_placements = [tuple(p) for p in placements]
    current_volume = sum(p[4]*p[5]*p[6] for p in current_placements if p[0]==0)
    best_placements = [tuple(p) for p in current_placements]
    best_volume = current_volume
    
    placed_idx = [i for i, p in enumerate(current_placements) if p[0] == 0]
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        if len(placed_idx) < 2:
            break
        
        # Pick two random placed boxes
        i, j = rng.sample(placed_idx, 2)
        
        pi = current_placements[i]
        pj = current_placements[j]
        
        wi, hi, di = boxes[i]
        wj, hj, dj = boxes[j]
        
        rots_i = _rotations(wi, hi, di)
        rots_j = _rotations(wj, hj, dj)
        
        # Try swapping: put box i at j's position, box j at i's position
        # First, remove both
        current_placements[i] = (-1, 0, 0, 0, wi, hi, di)
        current_placements[j] = (-1, 0, 0, 0, wj, hj, dj)
        
        placed_list = [tuple(p[1:7]) for p in current_placements if p[0] == 0]
        
        # Try to place box i at a position near where j was
        pos_i = (pj[1], pj[2], pj[3])
        pos_j = (pi[1], pi[2], pi[3])
        
        found_i = False
        for (rw, rh, rd) in rots_i:
            if _fits((rw, rh, rd), pos_i, container) and \
               _no_overlap((rw, rh, rd), pos_i, placed_list):
                current_placements[i] = (0, pos_i[0], pos_i[1], pos_i[2], rw, rh, rd)
                found_i = True
                break
        
        if found_i:
            placed_list = [tuple(p[1:7]) for p in current_placements if p[0] == 0]
        
        found_j = False
        for (rw, rh, rd) in rots_j:
            if _fits((rw, rh, rd), pos_j, container) and \
               _no_overlap((rw, rh, rd), pos_j, placed_list):
                current_placements[j] = (0, pos_j[0], pos_j[1], pos_j[2], rw, rh, rd)
                found_j = True
                break
        
        if found_i and found_j:
            new_volume = sum(p[4]*p[5]*p[6] for p in current_placements if p[0]==0)
            if new_volume > best_volume:
                best_volume = new_volume
                best_placements = [tuple(p) for p in current_placements]
        else:
            # Revert
            current_placements[i] = pi
            current_placements[j] = pj
    
    return best_placements, best_volume

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    min_dims = [min(b) for b in boxes]
    sum_dims = [sum(b) for b in boxes]
    surface_areas = [2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]) for b in boxes]
    
    # Phase 1: Generate diverse initial solutions
    strategies = []
    
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    strategies.append(('eap', vol_order, None, None))
    
    maxdim_order = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    strategies.append(('eap', maxdim_order, None, None))
    
    mindim_order = sorted(range(n), key=lambda i: min_dims[i])
    strategies.append(('eap', mindim_order, None, None))
    
    sumdim_order = sorted(range(n), key=lambda i: sum_dims[i], reverse=True)
    strategies.append(('eap', sumdim_order, None, None))
    
    sa_order = sorted(range(n), key=lambda i: surface_areas[i], reverse=True)
    strategies.append(('eap', sa_order, None, None))
    
    mixed1 = sorted(range(n), key=lambda i: (volumes[i], max_dims[i]), reverse=True)
    strategies.append(('eap', mixed1, None, None))
    
    def score_sum(pos, dim):
        return pos[0] + pos[1] + pos[2]
        
    def score_max(pos, dim):
        return max(pos)
        
    def score_z_sum(pos, dim):
        return (pos[2], pos[0] + pos[1])
        
    def score_corner(pos, dim):
        return (pos[0] * W + pos[1] * H + pos[2] * D)
        
    def score_min_dist(pos, dim):
        return pos[0] + pos[1] + pos[2]
        
    strategies.append(('eap', vol_order, score_sum, None))
    strategies.append(('eap', vol_order, score_max, None))
    strategies.append(('eap', vol_order, score_z_sum, None))
    strategies.append(('eap', vol_order, score_corner, None))
    
    rngs = [random.Random(i) for i in range(10)]
    for r in rngs:
        strategies.append(('eap', vol_order, None, r))
        
    rng_main = random.Random(42)
    for _ in range(10):
        order = list(range(n))
        rng_main.shuffle(order)
        strategies.append(('eap', order, None, random.Random(rng_main.randint(1, 1000))))
    
    # Add shelf and layered strategies
    strategies.append(('shelf', vol_order, None, None))
    strategies.append(('layered', vol_order, None, None))
    
    best_placements = None
    best_volume = -1
    
    for algo, order, scorer, rng in strategies:
        if time.time() > deadline - 50:
            break
        
        if algo == 'eap':
            placements, vol = _try_place_eap(boxes, container, order, deadline - 50, scorer, rng)
        elif algo == 'shelf':
            placements, vol = _shelf_place(boxes, container, order, deadline - 50, rng)
        elif algo == 'layered':
            placements, vol = _layered_place(boxes, container, order, deadline - 50, rng)
        else:
            continue
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume:
                break
    
    if best_placements is None:
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
        best_placements = placements
        best_volume = sum(p[4]*p[5]*p[6] for p in placements if p[0]==0)

    # Phase 2: Local search with relocation
    def get_placed_list(placements):
        placed = []
        for i, p in enumerate(placements):
            if p[0] == 0:
                placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        return placed
    
    def try_insert_box(box_idx, placements, container, rng=None, shuffle_candidates=False):
        w, h, d = boxes[box_idx]
        rots = _rotations(w, h, d)
        placed = get_placed_list(placements)
        candidates = _extreme_points(placed, container)
        
        if rng and shuffle_candidates and len(candidates) > 1:
            rng.shuffle(candidates)
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                return (0, cx, cy, cz, rw, rh, rd)
        return None
    
    # Local search procedure
    def local_search(initial_placements, rng, deadline, max_rounds=2):
        current_placements = [tuple(p) for p in initial_placements]
        current_volume = sum(p[4]*p[5]*p[6] for p in current_placements if p[0]==0)
        ls_best_placements = [tuple(p) for p in current_placements]
        ls_best_volume = current_volume
        
        for round_num in range(max_rounds):
            improved = False
            
            # Try to insert unplaced boxes
            unplaced = [i for i, p in enumerate(current_placements) if p[0] == -1]
            for idx in unplaced:
                if time.time() > deadline:
                    break
                new_pl = try_insert_box(idx, current_placements, container, rng)
                if new_pl:
                    vol = new_pl[4]*new_pl[5]*new_pl[6]
                    current_placements[idx] = new_pl
                    current_volume += vol
                    improved = True
            
            if current_volume > ls_best_volume:
                ls_best_volume = current_volume
                ls_best_placements = [tuple(p) for p in current_placements]
            
            if not improved:
                break
            
            # Try relocating placed boxes
            placed_idx = [i for i, p in enumerate(current_placements) if p[0] == 0]
            rng.shuffle(placed_idx)
            for idx in placed_idx:
                if time.time() > deadline:
                    break
                
                old_pl = current_placements[idx]
                old_vol = old_pl[4]*old_pl[5]*old_pl[6]
                
                current_placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                current_volume -= old_vol
                
                new_pl = try_insert_box(idx, current_placements, container, rng, shuffle_candidates=True)
                
                if new_pl:
                    new_vol = new_pl[4]*new_pl[5]*new_pl[6]
                    current_placements[idx] = new_pl
                    current_volume += new_vol
                    
                    if current_volume > ls_best_volume:
                        ls_best_volume = current_volume
                        ls_best_placements = [tuple(p) for p in current_placements]
                else:
                    current_placements[idx] = old_pl
                    current_volume += old_vol
            
            if time.time() > deadline:
                break
        
        return ls_best_placements, ls_best_volume
    
    # Run multiple local searches
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 15:
            break
        rng_ls = random.Random(seed)
        ls_placements, ls_volume = local_search(best_placements, rng_ls, deadline, max_rounds=2)
        
        if ls_volume > best_volume:
            best_volume = ls_volume
            best_placements = ls_placements
            if best_volume >= container_volume:
                break
    
    # Phase 3: Simulated Annealing refinement
    if time.time() < deadline - 10:
        rng_sa = random.Random(7)
        sa_placements, sa_volume = _simulated_annealing_refine(
            boxes, container, best_placements, deadline - 10, rng_sa)
        if sa_volume > best_volume:
            best_volume = sa_volume
            best_placements = sa_placements
    
    # Phase 4: Pairwise swaps
    if time.time() < deadline - 5:
        rng_swap = random.Random(13)
        swap_placements, swap_volume = _pairwise_swap(
            boxes, container, best_placements, deadline - 5, rng_swap, max_swaps=30)
        if swap_volume > best_volume:
            best_volume = swap_volume
            best_placements = swap_placements
    
    return best_placements
