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
        points = set()
        points.add((0, 0, 0))
        for (idx, x, y, z, w, h, d) in placed:
            # Add all 8 corners
            corners = [
                (x, y, z), (x+w, y, z), (x, y+h, z), (x+w, y+h, z),
                (x, y, z+d), (x+w, y, z+d), (x, y+h, z+d), (x+w, y+h, z+d)
            ]
            for c in corners:
                # Prune points outside container
                if 0 <= c[0] < W and 0 <= c[1] < H and 0 <= c[2] < D:
                    points.add(c)
        
        # Filter out points that are inside any placed box
        valid_points = []
        for p in points:
            if not is_point_inside_any_box(p, placed):
                valid_points.append(p)
        
        return valid_points

    def is_valid_placement(placed, idx, x, y, z, bw, bh, bd):
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

    def pack_with_order(order):
        placed = []
        # We maintain a dynamic set of candidate points.
        # For efficiency in a strict time limit, we recompute skyline points 
        # or maintain a list. Given N=60, N^2 overlap check is ~3600 ops per box, 
        # and skyline points are ~O(N). 
        # Let's use a simple list of candidates that we update.
        
        candidates = [(0,0,0)]
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_score = float('inf')
            
            # Try all orientations
            for (bw, bh, bd) in orient_cache[idx]:
                # Try all current candidates
                # Note: candidates might contain points that are now invalid due to new placements?
                # No, we only ADD points when we place a box. Existing points remain valid 
                # unless they are inside a NEW box. But we check is_valid_placement anyway.
                # However, the list of candidates can grow. 
                # To keep it small, we can prune candidates that are inside the newly placed box 
                # in the next iteration, or just let is_valid_placement handle it.
                # Given N=60, the number of corners is at most 8*60 = 480. This is small.
                
                for (cx, cy, cz) in candidates:
                    if is_valid_placement(placed, idx, cx, cy, cz, bw, bh, bd):
                        # Score: prefer positions that are "lower" in the container
                        # Standard BFD uses (z, y, x) or similar to pack tightly.
                        # Let's use a weighted sum.
                        score = 10000 * cz + 100 * cy + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, bw, bh, bd)
            
            if best_pos:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
                
                # Update candidates: add the 8 corners of the new box
                new_corners = [
                    (x, y, z), (x+bw, y, z), (x, y+bh, z), (x+bw, y+bh, z),
                    (x, y, z+bd), (x+bw, y, z+bd), (x, y+bh, z+bd), (x+bw, y+bh, z+bd)
                ]
                for nc in new_corners:
                    if 0 <= nc[0] < W and 0 <= nc[1] < H and 0 <= nc[2] < D:
                        # Check if this point is inside any EXISTING box (including the one just placed? No, it's a corner, so it's on the boundary)
                        # Actually, a corner of a placed box might be inside another placed box? No, boxes don't overlap.
                        # But a corner of box A might be inside box B if they are adjacent? No, if they are adjacent, the corner is on the boundary.
                        # If they are not adjacent, the corner is either inside or outside.
                        # If it's inside another box, it's not a valid starting point for a NEW box (it would overlap).
                        # So we should prune it.
                        if not is_point_inside_any_box(nc, placed):
                            candidates.append(nc)
                
                # Optional: Prune candidates that are now inside the newly placed box?
                # The newly placed box is in `placed`. 
                # Any candidate point that is inside the NEW box is invalid.
                # Since we just added the box, we can filter `candidates` to remove points inside it.
                # This keeps the list small.
                new_box = placed[-1]
                nx, ny, nz, nw, nh, nd = new_box[1], new_box[2], new_box[3], new_box[4], new_box[5], new_box[6]
                filtered_candidates = []
                for c in candidates:
                    if not (nx <= c[0] < nx + nw and ny <= c[1] < ny + nh and nz <= c[2] < nz + nd):
                        filtered_candidates.append(c)
                candidates = filtered_candidates

        return placed

    # --- Search Strategy ---
    
    indexed = list(range(n))
    
    # 1. Generate initial diverse orders
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    order_sum = sorted(indexed, key=lambda i: sum(boxes[i]), reverse=True)
    
    # Aspect ratio (cubic first)
    def aspect_ratio(i):
        w, h, d = boxes[i]
        return max(w, h, d) / min(w, h, d)
    order_aspect = sorted(indexed, key=aspect_ratio)
    
    # Random orders
    random_orders = []
    for _ in range(10):
        rand_order = indexed[:]
        random.shuffle(rand_order)
        random_orders.append(rand_order)
        
    all_orders = [order_vol, order_max, order_sum, order_aspect] + random_orders
    
    best_placed = []
    best_vol = 0
    best_order = None
    
    start_time = time.time()
    
    # Phase 1: Greedy BFD with multiple orders
    for order in all_orders:
        if time.time() > deadline:
            break
        p = pack_with_order(order)
        v = sum(p[4]*p[5]*p[6] for p in p)
        if v > best_vol:
            best_vol = v
            best_placed = p[:]
            best_order = order[:]
            if len(p) == n:
                break # Can't do better than 100%

    # Phase 2: Local Search on the Best Order
    # If we didn't pack everything, try to improve the order using 2-opt / 3-opt swaps.
    # We reconstruct the packing from the order.
    # The "state" is the order of boxes.
    # We evaluate an order by running pack_with_order.
    # This is expensive (O(N^2 * candidates)). With N=60, it's fast enough.
    
    if best_order and len(best_placed) < n:
        current_order = best_order[:]
        current_vol = best_vol
        no_improve_count = 0
        max_no_improve = 50 # Stop if we hit a plateau
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Randomly select two indices to swap
            i, j = random.sample(range(n), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            p = pack_with_order(new_order)
            v = sum(p[4]*p[5]*p[6] for p in p)
            
            if v > current_vol:
                current_vol = v
                current_order = new_order
                best_vol = v
                best_placed = p[:]
                no_improve_count = 0
                if len(p) == n:
                    break
            else:
                no_improve_count += 1
                
            # Also try 3-opt occasionally
            if random.random() < 0.1:
                i, j, k = random.sample(range(n), 3)
                new_order = current_order[:]
                # Rotate 3 elements
                new_order[i], new_order[j], new_order[k] = new_order[k], new_order[i], new_order[j]
                p = pack_with_order(new_order)
                v = sum(p[4]*p[5]*p[6] for p in p)
                if v > current_vol:
                    current_vol = v
                    current_order = new_order
                    best_vol = v
                    best_placed = p[:]
                    no_improve_count = 0
                    if len(p) == n:
                        break
                else:
                    no_improve_count += 1

    # Finalize result
    result = [None] * n
    for p in best_placed:
        idx, x, y, z, w, h, d = p
        result[idx] = (0, x, y, z, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return result

# EVOLVE-BLOCK-END
