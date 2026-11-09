import time
import random
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
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

def _try_place_sa(boxes, container, order_idx, deadline, rng):
    """Simulated Annealing approach."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    # Initial greedy construction using EAP logic for speed
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _extreme_points(placed, container)
        
        best_pos = None
        best_dim = None
        best_score = None
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                # Heuristic score: prefer bottom, then left, then back
                score = (cz, cy, cx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
        
        if best_pos is not None:
            placed.append((best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2]))
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
            
    # Track placed boxes indices for SA
    # placed_list contains tuples (x, y, z, w, h, d)
    # We need to map placed boxes back to original box indices to allow swapping/replacement
    # For simplicity in SA, we treat 'placed' as a list of active boxes.
    # We will maintain a list of (original_index, x, y, z, w, h, d)
    
    active_boxes = []
    for i in range(len(boxes)):
        if placements[i][0] == 0:
            active_boxes.append((i, placements[i][1], placements[i][2], placements[i][3], placements[i][4], placements[i][5], placements[i][6]))
            
    def calc_volume(active):
        return sum(w*h*d for (_, _, _, w, h, d) in active)
    
    current_vol = calc_volume(active_boxes)
    best_vol = current_vol
    best_active = [b for b in active_boxes]
    
    # SA Parameters
    T = 100.0 # Initial temperature (arbitrary units, scaled by volume diff)
    alpha = 0.999 # Cooling rate
    min_T = 0.1
    steps = 0
    max_steps = 10000 # Hard cap on steps to avoid infinite loops if time check is slow
    
    while time.time() < deadline and T > min_T and steps < max_steps:
        steps += 1
        
        # Generate move:
        # 1. Pick a random box to remove
        # 2. Pick a random box to add (from unplaced or placed? Usually add from unplaced to improve fill)
        # Let's try: Remove a random placed box, then try to place a random unplaced box (or the same one in a new spot)
        
        if not active_boxes:
            break
            
        remove_idx = rng.randint(0, len(active_boxes) - 1)
        box_to_remove = active_boxes[remove_idx]
        
        # Find an unplaced box to try to add
        unplaced_indices = [i for i in range(len(boxes)) if placements[i][0] == -1]
        
        if not unplaced_indices:
            # All boxes are placed? Just try to move one?
            # For now, if all placed, we are done or stuck.
            # Actually, if all placed, we might still want to optimize arrangement? 
            # But volume is fixed if all are placed. So we can stop.
            break
            
        add_idx = rng.choice(unplaced_indices)
        
        # Try to place add_idx
        w, h, d = boxes[add_idx]
        rots = _rotations(w, h, d)
        
        # Generate new active list
        new_active = [b for j, b in enumerate(active_boxes) if j != remove_idx]
        
        # Try to find a spot for add_idx in new_active
        candidates = _extreme_points([(b[1], b[2], b[3], b[4], b[5], b[6]) for b in new_active], container)
        
        found_spot = False
        new_pos = None
        new_dim = None
        
        # Randomize candidate order to avoid bias
        rng.shuffle(candidates)
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), [(b[1], b[2], b[3], b[4], b[5], b[6]) for b in new_active]):
                    continue
                new_pos = (cx, cy, cz)
                new_dim = (rw, rh, rd)
                found_spot = True
                break
            if found_spot:
                break
        
        if found_spot:
            # Calculate new volume
            # Volume change: + vol(add) - vol(remove)
            vol_add = w * h * d
            vol_remove = box_to_remove[4] * box_to_remove[5] * box_to_remove[6]
            delta_vol = vol_add - vol_remove
            
            if delta_vol > 0:
                accept = True
            else:
                # Boltzmann probability
                # Scale T to volume units? 
                # Let's use a relative temperature or just accept if random < exp(delta/T)
                # T needs to be in same units as volume.
                # Let's estimate max volume ~ container volume.
                # If T is 100, and delta is -10, exp(-0.1) is reasonable.
                prob = pow(2.71828, delta_vol / T)
                accept = rng.random() < prob
            
            if accept:
                # Update active list
                new_active.append((add_idx, new_pos[0], new_pos[1], new_pos[2], new_dim[0], new_dim[1], new_dim[2]))
                current_vol += delta_vol
                active_boxes = new_active
                
                # Update placements array temporarily? 
                # We only update the final placements at the end of SA or when best is found.
                # To save time, we don't update the global placements array inside the loop.
                
                if current_vol > best_vol:
                    best_vol = current_vol
                    best_active = [b for b in active_boxes]
        
        T *= alpha
        
    # Reconstruct placements from best_active
    final_placements = [None] * len(boxes)
    for i in range(len(boxes)):
        final_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
    for (idx, x, y, z, w, h, d) in best_active:
        final_placements[idx] = (0, x, y, z, w, h, d)
        
    return final_placements, best_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # Sorting strategies
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    
    best_placements = None
    best_volume = -1
    
    # Run SA with different initial orderings
    # SA is stochastic, so we run it multiple times with different seeds/orders
    rng = random.Random(42)
    
    # 1. SA with Volume order
    if time.time() < deadline:
        placements, vol = _try_place_sa(boxes, container, vol_order, deadline, rng)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
    
    # 2. SA with MaxDim order
    if time.time() < deadline:
        rng2 = random.Random(123)
        placements, vol = _try_place_sa(boxes, container, maxdim_order, deadline, rng2)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            
    # 3. Pure EAP fallback if SA failed or for comparison
    # (Keep a fast EAP as a safety net)
    if time.time() < deadline:
        W, H, D = container
        placed = []
        placements = [None] * len(boxes)
        for orig_idx in vol_order:
            w, h, d = boxes[orig_idx]
            rots = _rotations(w, h, d)
            candidates = _extreme_points(placed, container)
            best_pos = None
            best_dim = None
            best_score = None
            for (cx, cy, cz) in candidates:
                for (rw, rh, rd) in rots:
                    if not _fits((rw, rh, rd), (cx, cy, cz), container):
                        continue
                    if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                        continue
                    score = (cz, cy, cx)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_dim = (rw, rh, rd)
            if best_pos is not None:
                placed.append((best_pos[0], best_pos[1], best_pos[2], best_dim[0], best_dim[1], best_dim[2]))
                placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2], best_dim[0], best_dim[1], best_dim[2])
            else:
                placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
        for i in range(len(boxes)):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
        vol = sum(p[4]*p[5]*p[6] for p in placements if p[0]==0)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements

    if best_placements is None:
        # Fallback: simple shelf
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
        
    return best_placements
