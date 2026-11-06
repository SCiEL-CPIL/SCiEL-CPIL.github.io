import time
import random
import itertools
import math

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    # Safety margin: The evaluator kills at 150s, but the task says 120s limit.
    # We use 115s to be safe.
    deadline = time.time() + 115.0
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    
    # --- Grid-Based Packing Engine ---
    # We discretize the container into a grid of 1x1x1 cells (or larger if container is huge, 
    # but 60 boxes usually implies manageable dimensions).
    # If dimensions are too large for a full grid (e.g., > 100 in any dim), we fall back to 
    # a coarser grid or the previous skyline method. 
    # For now, assume standard sizes. If W*H*D > 1e6, we use a sparse representation or coarser grid.
    
    max_dim = max(W, H, D)
    # If the container is very large, a full grid is too slow/memory heavy.
    # We will use a sparse "occupied" set of grid cells for overlap checking, 
    # but iterate over "candidate anchor points" derived from box boundaries.
    # Actually, for "Best-Insertion", we want to try placing a box at every possible 
    # valid (x,y,z) that aligns with existing box boundaries or container walls.
    
    # To make "Best-Insertion" efficient, we maintain a list of "candidate points".
    # A candidate point is (x,y,z) such that a box placed there would be supported 
    # (bottom face on floor or top of another box) and fits in the container.
    
    def pack_best_insertion(order):
        """
        Greedy packing where we pick the 'best' position for the next box 
        based on a heuristic score (e.g., lowest z, then lowest y, then lowest x).
        This is equivalent to First-Fit-Decreasing if we process boxes in order 
        and pick the first valid spot. 
        However, 'Best-Insertion' usually implies looking at ALL valid spots and picking 
        the one that minimizes some cost function (like contact area or height).
        
        Here we use a hybrid: 
        1. Iterate through boxes in the given order.
        2. For each box, find ALL valid positions (x,y,z,orient) that fit.
        3. Pick the position with the lowest (z, y, x) to pack tightly to the bottom-left-front.
        
        To speed this up, we don't scan the whole grid. We scan candidate points 
        generated from the corners of existing boxes.
        """
        placed = [] # list of (idx, x, y, z, w, h, d)
        candidates = set([(0, 0, 0)])
        
        # We also need to check if a point is "supported". 
        # A box at (x,y,z) is supported if z==0 OR there is a box directly below 
        # covering the entire footprint (x, x+w) x (y, y+h) at some z' < z such that 
        # the gap is 0? No, standard packing allows floating if supported by multiple boxes?
        # Usually, "gravity" models require the bottom face to be fully supported.
        # However, for maximizing volume, we can allow "floating" if we just check overlap.
        # The task says "without overlap". It does NOT explicitly say "no floating".
        # But physically, boxes fall. The standard interpretation in these coding tasks 
        # is usually just "no overlap" and "inside container". 
        # If floating is allowed, the problem is easier. 
        # Let's assume "no overlap" is the only constraint.
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_score = float('inf')
            
            # Generate candidate points
            # Current candidates are corners of placed boxes + (0,0,0)
            # We also need to consider points that might be "inside" the container 
            # but not on a corner? 
            # If we only use corners, we might miss a spot where a box fits in a "hole" 
            # formed by 4 boxes, but the corner of the hole is not a corner of any single box?
            # No, the corners of the hole ARE corners of the surrounding boxes.
            # So the set of corners of all placed boxes + (0,0,0) is a sufficient 
            # set of candidate anchor points for axis-aligned boxes if we allow 
            # the box to start at any coordinate that aligns with a box edge.
            
            # Optimization: 
            # Instead of checking all candidates for all orientations, 
            # we can iterate candidates, and for each, check if it's valid.
            
            # To improve speed, we can limit the number of candidates we check.
            # But with N=60, candidates are at most 8*60 + 1 = 481.
            # Orientations are at most 6.
            # 481 * 6 = ~2886 checks per box.
            # Overlap check is O(N) = 60.
            # Total ops ~ 2886 * 60 * 60 (boxes) = ~10 million ops. 
            # This is fast in Python if optimized.
            
            # We will iterate candidates.
            # To ensure we find the "best" (lowest) position, we iterate all valid ones.
            
            # To speed up, we can sort candidates by (z, y, x) and stop early?
            # No, a higher z might be the only valid one. We must check all.
            
            # However, we can prune candidates that are clearly "bad" (high z) if we find a good one?
            # No, we want the absolute lowest z.
            
            for (cx, cy, cz) in candidates:
                # Check if this point is inside any placed box (invalid anchor)
                # If the anchor is inside a box, we can't start there.
                inside = False
                for (p_idx, px, py, pz, pw, ph, pd) in placed:
                    if px <= cx < px + pw and py <= cy < py + ph and pz <= cz < pz + pd:
                        inside = True
                        break
                if inside:
                    continue
                
                # Try all orientations
                for (bw, bh, bd) in orient_cache[idx]:
                    # Check bounds
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    # Check overlap
                    valid = True
                    for (p_idx, px, py, pz, pw, ph, pd) in placed:
                        if (cx < px + pw and cx + bw > px and
                            cy < py + ph and cy + bh > py and
                            cz < pz + pd and cz + bd > pz):
                            valid = False
                            break
                    
                    if valid:
                        # Score: Minimize z, then y, then x
                        score = 10000 * cz + 100 * cy + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, bw, bh, bd)
            
            if best_pos:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
                
                # Update candidates
                # Add 8 corners of the new box
                new_corners = [
                    (x, y, z), (x+bw, y, z), (x, y+bh, z), (x+bw, y+bh, z),
                    (x, y, z+bd), (x+bw, y, z+bd), (x, y+bh, z+bd), (x+bw, y+bh, z+bd)
                ]
                for nc in new_corners:
                    if 0 <= nc[0] < W and 0 <= nc[1] < H and 0 <= nc[2] < D:
                        candidates.add(nc)
                
                # Prune candidates that are now inside the newly placed box
                # This keeps the candidate list small
                filtered = set()
                for c in candidates:
                    if not (x <= c[0] < x + bw and y <= c[1] < y + bh and z <= c[2] < z + bd):
                        filtered.add(c)
                candidates = filtered
                
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
    for _ in range(5):
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
        T = 100.0
        T_min = 1.0
        cooling_rate = 0.995
        
        no_improve = 0
        max_no_improve = 100 # Plateau detection
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Generate neighbor: Swap two boxes in the order
            i, j = random.sample(range(n), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
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
                if random.random() < math.exp(delta / T):
                    current_vol = v
                    current_order = new_order
                else:
                    no_improve += 1
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # If we hit T_min and no improvement for a while, restart with higher T?
                # Or just stop if no_improve is high.
                # Let's just keep going until deadline or plateau.
                
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
