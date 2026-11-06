"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    deadline = time.time() + 115.0  # Safety margin for 120s limit

    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]

    # --- Packing Engine with Numpy Occupancy ---
    occupancy = np.zeros((W, H, D), dtype=np.uint8)

    def is_valid_placement_fast(x, y, z, bw, bh, bd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        if np.any(occupancy[z:z+bd, y:y+bh, x:x+bw]):
            return False
        return True

    def place_box(x, y, z, bw, bh, bd):
        occupancy[z:z+bd, y:y+bh, x:x+bw] = 1

    def pack_with_order(order, candidate_strategy='extreme'):
        occupancy[:] = 0
        placed = []
        candidates = [(0, 0, 0)]
        
        for idx in order:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[idx]
            best_pos = None
            best_score = float('inf')
            
            for (bw, bh, bd) in orient_cache[idx]:
                for (cx, cy, cz) in candidates:
                    if is_valid_placement_fast(cx, cy, cz, bw, bh, bd):
                        # BFD heuristic: minimize z, then y, then x
                        score = 10000 * cz + 100 * cy + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, bw, bh, bd)
            
            if best_pos:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
                place_box(x, y, z, bw, bh, bd)
                
                # Update candidates: add corners of the new box
                new_corners = [
                    (x, y, z), (x+bw, y, z), (x, y+bh, z), (x+bw, y+bh, z),
                    (x, y, z+bd), (x+bw, y, z+bd), (x, y+bh, z+bd), (x+bw, y+bh, z+bd)
                ]
                
                for nc in new_corners:
                    if 0 <= nc[0] < W and 0 <= nc[1] < H and 0 <= nc[2] < D:
                        if occupancy[nc[2], nc[1], nc[0]] == 0:
                            if nc not in candidates:
                                candidates.append(nc)
                
                # Prune occupied candidates
                new_candidates = [c for c in candidates if occupancy[c[2], c[1], c[0]] == 0]
                candidates = new_candidates

        return placed

    # --- Search Strategy ---
    indexed = list(range(n))
    
    # Generate diverse initial orders
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    order_sum = sorted(indexed, key=lambda i: sum(boxes[i]), reverse=True)
    
    def aspect_ratio(i):
        w, h, d = boxes[i]
        m = min(w, h, d)
        if m == 0: return 0
        return max(w, h, d) / m
    order_aspect = sorted(indexed, key=aspect_ratio)
    
    random_orders = []
    for _ in range(8):
        rand_order = indexed[:]
        random.shuffle(rand_order)
        random_orders.append(rand_order)
        
    all_orders = [order_vol, order_max, order_sum, order_aspect] + random_orders
    
    best_placed = []
    best_vol = 0
    best_order = None
    
    # Phase 1: Greedy BFD with multiple orders
    for order in all_orders:
        if time.time() > deadline:
            break
        p = pack_with_order(order)
        vol = sum(w*h*d for (_, x, y, z, w, h, d) in p)
        if vol > best_vol:
            best_vol = vol
            best_placed = p
            best_order = order
        if vol >= W*H*D:
            break

    # Phase 2: Simulated Annealing on permutations
    if time.time() < deadline - 5:
        current_order = best_order[:]
        current_vol = best_vol
        T = 100.0
        T_min = 1.0
        cooling = 0.995
        
        # Pre-compute a few static orders to jump to occasionally
        jump_orders = [order_vol, order_max, order_sum, order_aspect]
        
        step = 0
        while time.time() < deadline - 5 and T > T_min:
            step += 1
            
            # Occasionally jump to a known good heuristic order
            if random.random() < 0.05:
                candidate_order = random.choice(jump_orders)[:]
            else:
                # Perturb current order: swap, insert, or reverse segment
                candidate_order = current_order[:]
                move_type = random.choice(['swap', 'insert', 'reverse'])
                
                if move_type == 'swap':
                    i, j = random.randint(0, n-1), random.randint(0, n-1)
                    candidate_order[i], candidate_order[j] = candidate_order[j], candidate_order[i]
                elif move_type == 'insert':
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i != j:
                        item = candidate_order.pop(i)
                        candidate_order.insert(j, item)
                else:  # reverse
                    i, j = random.randint(0, n-1), random.randint(0, n-1)
                    if i > j:
                        i, j = j, i
                    candidate_order[i:j+1] = candidate_order[i:j+1][::-1]
            
            p = pack_with_order(candidate_order)
            vol = sum(w*h*d for (_, x, y, z, w, h, d) in p)
            
            # Acceptance criterion
            if vol > current_vol or random.random() < np.exp((vol - current_vol) / max(T, 1e-9)):
                current_order = candidate_order
                current_vol = vol
                if vol > best_vol:
                    best_vol = vol
                    best_placed = p
                    best_order = current_order[:]
            
            T *= cooling
    
    # Construct final result
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    for (idx, x, y, z, w, h, d) in best_placed:
        placements[idx] = (0, x, y, z, w, h, d)
        
    return placements
# EVOLVE-BLOCK-END
