import time
import random
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    deadline = time.time() + 115.0  # Safety margin for 150s limit
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]

    # --- 3D Occupancy Bitmap Packing Engine ---
    # The container is represented as a 3D numpy array.
    # A placement is valid if the sum of the occupancy in the target volume is 0.
    # This allows vectorized O(1) (amortized) overlap checks and fast candidate filtering.
    
    def pack_with_order(order):
        # occupancy array of shape (W, H, D)
        # Using int8 to keep memory low, though 1s are enough.
        occ = np.zeros((W, H, D), dtype=np.int8)
        placed = []
        
        # Candidate points: we maintain a set of (x, y, z) points.
        # To speed up, we can precompute a list of all points, but that's W*H*D which might be large.
        # Instead, we use the corner-expansion strategy but check validity using the bitmap.
        candidates = [(0, 0, 0)]
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_score = float('inf')
            
            for (bw, bh, bd) in orient_cache[idx]:
                # Check if box fits in container
                if bw > W or bh > H or bd > D:
                    continue
                
                for (cx, cy, cz) in candidates:
                    # Check bounds
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    # Vectorized overlap check
                    # If the sum of the sub-array is 0, it's empty.
                    # occ is (W, H, D). We want occ[cx:cx+bw, cy:cy+bh, cz:cz+bd]
                    # Note: numpy slicing is [x, y, z] corresponding to axes 0, 1, 2.
                    # Wait, standard convention is usually (x, y, z) or (z, y, x).
                    # Let's stick to (x, y, z) -> axis 0, 1, 2.
                    # occ[cx:cx+bw, cy:cy+bh, cz:cz+bd]
                    
                    # Optimization: if the box is large, this slice is large.
                    # For N=60 and typical container sizes, this should be fine.
                    # If container is huge (e.g., 100x100x100), the array is 1M elements.
                    # Slicing and summing is fast in C.
                    
                    # Check validity
                    if np.sum(occ[cx:cx+bw, cy:cy+bh, cz:cz+bd]) == 0:
                        # Score: prefer positions that are "lower" in the container
                        # Standard BFD uses (z, y, x) or similar.
                        # Let's use a weighted sum to pack tightly towards the origin.
                        score = 10000 * cz + 100 * cy + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, bw, bh, bd)
            
            if best_pos:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
                
                # Update occupancy
                occ[x:x+bw, y:y+bh, z:z+bd] = 1
                
                # Update candidates: add the 8 corners of the new box
                new_corners = [
                    (x, y, z), (x+bw, y, z), (x, y+bh, z), (x+bw, y+bh, z),
                    (x, y, z+bd), (x+bw, y, z+bd), (x, y+bh, z+bd), (x+bw, y+bh, z+bd)
                ]
                for nc in new_corners:
                    if 0 <= nc[0] < W and 0 <= nc[1] < H and 0 <= nc[2] < D:
                        # Check if this point is inside any EXISTING box (including the one just placed? No, it's a corner).
                        # A corner of a placed box might be inside another placed box? No, boxes don't overlap.
                        # But a corner of box A might be inside box B if they are adjacent? No, if they are adjacent, the corner is on the boundary.
                        # If they are not adjacent, the corner is either inside or outside.
                        # If it's inside another box, it's not a valid starting point for a NEW box (it would overlap).
                        # So we should prune it.
                        # Using the bitmap: if occ[nc] == 1, it's inside a box.
                        # Note: nc is a point (x, y, z). If occ[x, y, z] is 1, it's occupied.
                        if occ[nc[0], nc[1], nc[2]] == 0:
                            candidates.append(nc)
                
                # Prune candidates that are now inside the newly placed box?
                # The newly placed box is marked in occ.
                # Any candidate point that is inside the NEW box is invalid.
                # We can filter `candidates` to remove points where occ[x, y, z] == 1.
                # This keeps the list small.
                # However, iterating through all candidates and checking occ is O(C).
                # We can do this more efficiently by just checking the new box bounds.
                nx, ny, nz, nw, nh, nd = x, y, z, bw, bh, bd
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
