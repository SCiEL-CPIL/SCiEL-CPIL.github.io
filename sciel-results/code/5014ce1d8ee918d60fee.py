import time
import random
import itertools

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    deadline = time.time() + 115.0  # Safety margin for 120s limit

    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    total_vol = sum(box_vols)

    # --- Skyline-Based Packing Engine ---
    
    # State: list of placed boxes (idx, x, y, z, w, h, d)
    # Skyline: list of candidate points (x, y, z) where a new box could start.
    # Initially just (0,0,0).
    # After placing a box, we add its 8 corners to the candidate pool, 
    # but we prune any point that is "inside" an existing box.
    
    def is_point_inside_any_box(point, placed):
        px, py, pz = point
        for (idx, x, y, z, w, h, d) in placed:
            if x <= px < x + w and y <= py < y + h and z <= pz < z + d:
                return True
        return False

    def get_skyline_points(placed):
        points = {(0, 0, 0)}
        for (idx, x, y, z, w, h, d) in placed:
            # Generate 8 corners
            for dx in [0, w]:
                for dy in [0, h]:
                    for dz in [0, d]:
                        px, py, pz = x + dx, y + dy, z + dz
                        if 0 <= px < W and 0 <= py < H and 0 <= pz < D:
                            # Only add if not inside any existing box
                            if not is_point_inside_any_box((px, py, pz), placed):
                                points.add((px, py, pz))
        return list(points)

    def is_valid(placed, idx, x, y, z, bw, bh, bd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        
        # Check overlap with all placed boxes
        for (p_idx, px, py, pz, pw, ph, pd) in placed:
            if p_idx == idx:
                continue
            # Check overlap conditions
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        return True

    def get_volume(placed):
        return sum(p[4]*p[5]*p[6] for p in placed)

    # Greedy packing using Skyline points
    def greedy_pack(order, use_skyline=True):
        placed = []
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_score = float('inf')
            
            # Try all orientations
            for (bw, bh, bd) in orient_cache[idx]:
                if use_skyline:
                    candidates = get_skyline_points(placed)
                else:
                    # Fallback to simpler candidate generation if needed, 
                    # but Skyline is generally better for 3D
                    candidates = [(0,0,0)]
                    for (_, px, py, pz, pw, ph, pd) in placed:
                        candidates.append((px + pw, py, pz))
                        candidates.append((px, py + ph, pz))
                        candidates.append((px, py, pz + pd))
                        candidates.append((px + pw, py + ph, pz))
                        candidates.append((px + pw, py, pz + pd))
                        candidates.append((px, py + ph, pz + pd))
                        candidates.append((px + pw, py + ph, pz + pd))

                for (cx, cy, cz) in candidates:
                    if is_valid(placed, idx, cx, cy, cz, bw, bh, bd):
                        # Score: prefer lower z, then y, then x
                        # Weighting can be tuned, but standard (z, y, x) is robust
                        score = 1.0 * cz + 1.0 * cy + 1.0 * cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, bw, bh, bd)
            
            if best_pos:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
        
        return placed

    # Generate diverse ordering strategies
    indexed = list(range(n))
    
    # 1. Volume descending
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    # 2. Max dimension descending
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    # 3. Sum of dimensions descending
    order_sum = sorted(indexed, key=lambda i: sum(boxes[i]), reverse=True)
    # 4. Aspect ratio (most cubic first)
    def aspect_ratio(i):
        w, h, d = boxes[i]
        return max(w, h, d) / min(w, h, d)
    order_aspect = sorted(indexed, key=aspect_ratio)
    # 5. Random shuffles
    random_orders = []
    for _ in range(10):
        rand_order = indexed[:]
        random.shuffle(rand_order)
        random_orders.append(rand_order)
    
    all_orders = [order_vol, order_max, order_sum, order_aspect] + random_orders
    
    best_placed = []
    best_vol = 0
    
    # Time check inside loop
    start_time = time.time()
    
    for order in all_orders:
        if time.time() > deadline:
            break
            
        p = greedy_pack(order, use_skyline=True)
        v = get_volume(p)
        
        if v > best_vol:
            best_vol = v
            best_placed = p[:]
            
            # If we've packed everything, we can stop early? 
            # No, because we might find a "better" packing in terms of volume if some boxes were left out?
            # Actually, if all boxes are packed, volume is fixed (sum of all box vols). 
            # Since we maximize volume, if we pack all, that's the max possible volume.
            if len(p) == n:
                break

    # If we still have time, do a second pass with more random orders or local search?
    # Let's add a simple "swap" local search on the best order found?
    # No, let's just run more random orders if time permits.
    
    elapsed = time.time() - start_time
    if elapsed < 100: # If we finished fast, do more random restarts
        remaining_budget = 110.0 - elapsed
        # Estimate time per greedy run
        time_per_run = elapsed / len(all_orders) if len(all_orders) > 0 else 0.1
        max_extra_runs = int(remaining_budget / time_per_run) if time_per_run > 0 else 50
        max_extra_runs = min(max_extra_runs, 200) # Cap to avoid infinite loop if time_per_run is tiny
        
        for _ in range(max_extra_runs):
            if time.time() > deadline:
                break
            rand_order = indexed[:]
            random.shuffle(rand_order)
            p = greedy_pack(rand_order, use_skyline=True)
            v = get_volume(p)
            if v > best_vol:
                best_vol = v
                best_placed = p[:]
                if len(p) == n:
                    break

    # Finalize result
    result = [None] * n
    for p in best_placed:
        idx, x, y, z, w, h, d = p
        result[idx] = (0, x, y, z, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return result
