import time
import random
import itertools
import math
import numpy as np

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    # Safety margin: Evaluator kills at 150s. Use 110s to be safe.
    deadline = time.time() + 110.0
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    
    # --- Vectorized Grid-Based Packing Engine ---
    # Use a 3D numpy array for occupancy. 
    # Dimensions must be manageable. If too large, we might fall back, 
    # but typical 60-box instances likely fit in a reasonable grid (e.g., < 100^3).
    # If W*H*D is huge, numpy allocation might fail or be slow. 
    # Let's assume standard sizes. If W*H*D > 5e6, we might need a different strategy, 
    # but for now, let's try to allocate. 
    # To be safe against memory errors, we can cap the grid size or use a fallback.
    # However, the prompt implies standard sizes. 
    # Let's check size.
    total_cells = W * H * D
    if total_cells > 10_000_000:
        # Fallback to a simpler list-based method if grid is too big? 
        # Or just use a sparse set? 
        # For this specific strategy change, we rely on the grid. 
        # If it's too big, we might timeout on allocation. 
        # Let's assume it fits. If not, we can't easily vectorize without sparse structures.
        # Given the "60 boxes" constraint, containers are usually not 1000x1000x1000.
        pass

    occ = np.zeros((W, H, D), dtype=np.uint8)
    # Pre-compute slices for faster access? No, dynamic slices are fast enough in numpy.
    
    def pack_best_insertion(order):
        """
        Greedy packing using numpy occupancy grid.
        """
        placed = []
        # Candidate points: corners of placed boxes + (0,0,0)
        # We maintain a set of candidate (x,y,z) points.
        candidates = [(0, 0, 0)]
        # To avoid duplicates in candidates, we can use a set, but checking membership is O(1).
        # However, iterating a set is fine.
        
        # We need to be careful: if we add corners, we might add many.
        # Max candidates ~ 8 * N + 1 = 481.
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_score = float('inf')
            
            # Iterate candidates
            # Note: We should iterate over a snapshot or handle additions carefully.
            # Since we only add to candidates after placing a box, and we pick the best for current box,
            # the candidate set is static during the inner loop for this box.
            
            for (cx, cy, cz) in candidates:
                # Quick bounds check for anchor
                if cx < 0 or cy < 0 or cz < 0:
                    continue
                if cx >= W or cy >= H or cz >= D:
                    continue
                
                # Try all orientations
                for (bw, bh, bd) in orient_cache[idx]:
                    # Check bounds
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    # Vectorized overlap check
                    # Slice the occupancy grid
                    # If the slice sum is 0, it's free.
                    # occ is uint8, so sum is fast.
                    # Note: slicing creates a view, summing is O(volume).
                    # To optimize, we can check the min/max of the slice? No, sum is fine for small boxes.
                    # Actually, `occ[cx:cx+bw, cy:cy+bh, cz:cz+bd].sum() == 0`
                    
                    # Optimization: Check if the box fits in the free space.
                    # If the box is small, this is very fast.
                    if occ[cx:cx+bw, cy:cy+bh, cz:cz+bd].sum() == 0:
                        # Valid placement
                        # Score: Minimize z, then y, then x (pack to bottom-left-front)
                        # We can also add a penalty for "floating" if we cared, but task allows it.
                        # Let's stick to (z, y, x) minimization.
                        score = 10000 * cz + 100 * cy + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, bw, bh, bd)
            
            if best_pos:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
                
                # Update occupancy
                occ[x:x+bw, y:y+bh, z:z+bd] = 1
                
                # Update candidates
                # Add 8 corners of the new box
                new_corners = [
                    (x, y, z), (x+bw, y, z), (x, y+bh, z), (x+bw, y+bh, z),
                    (x, y, z+bd), (x+bw, y, z+bd), (x, y+bh, z+bd), (x+bw, y+bh, z+bd)
                ]
                for nc in new_corners:
                    if 0 <= nc[0] < W and 0 <= nc[1] < H and 0 <= nc[2] < D:
                        candidates.append(nc)
                
                # Prune candidates that are now occupied? 
                # If a candidate point is inside an occupied cell, it can't be an anchor for a new box 
                # (since the box would overlap immediately at 0,0,0 relative to anchor? 
                # Wait, if anchor is (x,y,z) and occ[x,y,z]==1, then any box placed there overlaps.
                # So we can remove candidates that are inside occupied cells.
                # This keeps the list small.
                # However, checking `occ[cx, cy, cz] == 0` is O(1).
                # Let's filter the list.
                # Note: We must be careful not to remove (0,0,0) if it's occupied? 
                # If (0,0,0) is occupied, we can't place anything there.
                
                # Efficient pruning:
                # candidates = [c for c in candidates if occ[c[0], c[1], c[2]] == 0]
                # This is O(K) where K is num candidates. K is small.
                candidates = [c for c in candidates if occ[c[0], c[1], c[2]] == 0]
                
                # Also, we might have duplicates. We can convert to set then back to list?
                # Or just leave duplicates? Duplicates cause redundant checks.
                # Let's deduplicate occasionally or always.
                # Since we add 8 per box, and prune, size is manageable.
                # Let's use a set for unique candidates to avoid redundant checks in the next iteration.
                # But we need to iterate.
                # Converting list to set to list is O(K).
                candidates = list(set(candidates))

        return placed

    # --- Search Strategy: Simulated Annealing on Order ---
    
    indexed = list(range(n))
    
    # Initial diverse orders
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    order_sum = sorted(indexed, key=lambda i: sum(boxes[i]), reverse=True)
    
    def aspect_ratio(i):
        w, h, d = boxes[i]
        return max(w, h, d) / min(w, h, d)
    order_aspect = sorted(indexed, key=aspect_ratio)
    
    all_orders = [order_vol, order_max, order_sum, order_aspect]
    
    # Add a few random orders
    for _ in range(10):
        rand_order = indexed[:]
        random.shuffle(rand_order)
        all_orders.append(rand_order)
        
    best_placed = []
    best_vol = 0
    best_order = None
    
    # Evaluate initial orders
    for order in all_orders:
        if time.time() > deadline:
            break
        p = pack_best_insertion(order)
        v = sum(p[4]*p[5]*p[6] for p in p)
        if v > best_vol:
            best_vol = v
            best_placed = p[:]
            best_order = order[:]
            if len(p) == n:
                break # Perfect pack
    
    # If we have time, run Simulated Annealing
    if best_order and len(best_placed) < n and time.time() < deadline:
        current_order = best_order[:]
        current_vol = best_vol
        
        # SA Parameters
        # We can afford more iterations now due to speedup.
        T = 50.0
        T_min = 0.5
        cooling_rate = 0.99
        
        no_improve = 0
        max_no_improve = 200 # Plateau detection
        
        # We can also try different neighborhood moves: Swap, Insert, Reverse
        while time.time() < deadline and no_improve < max_no_improve:
            # Generate neighbor
            move_type = random.random()
            new_order = current_order[:]
            
            if move_type < 0.5:
                # Swap
                i, j = random.sample(range(n), 2)
                new_order[i], new_order[j] = new_order[j], new_order[i]
            elif move_type < 0.8:
                # Insert: Remove i, insert at j
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                item = new_order.pop(i)
                new_order.insert(j, item)
            else:
                # Reverse segment
                i, j = random.sample(range(n), 2)
                if i > j: i, j = j, i
                new_order[i:j+1] = reversed(new_order[i:j+1])
            
            p = pack_best_insertion(new_order)
            v = sum(p[4]*p[5]*p[6] for p in p)
            
            delta = v - current_vol
            
            if delta > 0:
                # Accept
                current_vol = v
                current_order = new_order
                no_improve = 0
                if v > best_vol:
                    best_vol = v
                    best_placed = p[:]
                    if len(p) == n:
                        break
            else:
                # Metropolis Criterion
                if T > 0:
                    # Avoid exp overflow/underflow
                    ratio = delta / T
                    if ratio > -50: # exp(-50) is very small
                        if random.random() < math.exp(ratio):
                            current_vol = v
                            current_order = new_order
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
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
