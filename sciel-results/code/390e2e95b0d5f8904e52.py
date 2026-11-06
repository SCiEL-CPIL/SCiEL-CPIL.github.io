"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    deadline = time.time() + 115.0  # safety margin vs 150s kill
    container_vol = W * H * D
    
    # Precompute unique orientations for each box, sorted by priority
    # Priority: prefer orientations that minimize max dimension (more compact)
    orient = []
    for w, h, d in boxes:
        opts = set()
        for p in itertools.permutations((w, h, d)):
            opts.add(p)
        # Sort orientations: prefer smaller max dim, then smaller sum
        orient.append(sorted(opts, key=lambda o: (max(o), sum(o))))
    
    box_vol = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    total_vol = sum(box_vol)
    
    def try_pack(order, time_limit):
        """
        Fast 3D pack using anchor-based placement with early termination.
        Returns (filled_volume, placements_list) where placements_list is a list of (box_idx, x, y, z, w, h, d)
        """
        placed = []  # list of (x, y, z, w, h, d, box_idx)
        filled = 0
        
        # Anchors: (x, y, z) - potential bottom-front-left corners
        anchors = [(0, 0, 0)]
        anchor_set = {(0, 0, 0)}
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_key = None
            
            for (ow, oh, od) in orient[idx]:
                if ow > W or oh > H or od > D:
                    continue
                
                # Check anchors - sort by priority for early termination
                # We'll collect valid positions and pick the best
                valid_positions = []
                
                for (ax, ay, az) in anchors:
                    if ax + ow > W or ay + oh > H or az + od > D:
                        continue
                    
                    # Overlap check
                    ok = True
                    for (px, py, pz, pw, ph, pd, pidx) in placed:
                        if not (ax + ow <= px or px + pw <= ax or
                                ay + oh <= py or py + ph <= ay or
                                az + od <= pz or pz + pd <= az):
                            ok = False
                            break
                    
                    if ok:
                        key = (az, ay, ax)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (ax, ay, az, ow, oh, od)
                        if best_key == (0, 0, 0):
                            break
                
                if best_key == (0, 0, 0):
                    break
            
            if best_pos is not None:
                bx, by, bz, bw, bh, bd = best_pos
                placed.append((bx, by, bz, bw, bh, bd, idx))
                filled += bw * bh * bd
                
                # Generate new anchors from the placed box
                new_anchors = []
                if bz + bd <= D:
                    new_anchors.append((bx, by, bz + bd))
                if bx + bw <= W:
                    new_anchors.append((bx + bw, by, bz))
                if by + bh <= H:
                    new_anchors.append((bx, by + bh, bz))
                if bx + bw <= W and by + bh <= H:
                    new_anchors.append((bx + bw, by + bh, bz))
                if bx + bw <= W and bz + bd <= D:
                    new_anchors.append((bx + bw, by, bz + bd))
                if by + bh <= H and bz + bd <= D:
                    new_anchors.append((bx, by + bh, bz + bd))
                if bx + bw <= W and by + bh <= H and bz + bd <= D:
                    new_anchors.append((bx + bw, by + bh, bz + bd))
                
                for na in new_anchors:
                    if na not in anchor_set:
                        anchor_set.add(na)
                        anchors.append(na)
                        
                # Early termination if container is nearly full
                if filled >= container_vol * 0.95:
                    break
                
                # Time check
                if time.time() > time_limit:
                    break
        
        return filled, placed
    
    def make_orders():
        orders = []
        # 1. Descending volume
        orders.append(sorted(range(n), key=lambda i: -box_vol[i]))
        # 2. Ascending volume
        orders.append(sorted(range(n), key=lambda i: box_vol[i]))
        # 3. Largest min dimension
        orders.append(sorted(range(n), key=lambda i: -min(boxes[i])))
        # 4. Largest max dimension
        orders.append(sorted(range(n), key=lambda i: -max(boxes[i])))
        # 5. Descending surface area
        def surf(i):
            a, b, c = boxes[i]
            return 2 * (a * b + a * c + b * c)
        orders.append(sorted(range(n), key=lambda i: -surf(i)))
        # 6. Descending "compactness" (vol / max_dim)
        def compact(i):
            a, b, c = boxes[i]
            m = max(a, b, c)
            return (a * b * c) / m if m else 0
        orders.append(sorted(range(n), key=lambda i: -compact(i)))
        # 7. Descending aspect ratio balance (prefer cubical)
        def balance(i):
            a, b, c = sorted(boxes[i])
            if c == 0:
                return 0
            return (a * b) / c  # larger means more balanced
        orders.append(sorted(range(n), key=lambda i: -balance(i)))
        # 8. Descending sum of dimensions
        orders.append(sorted(range(n), key=lambda i: -(sum(boxes[i]))))
        return orders
    
    best_filled = -1
    best_placed = []
    
    base_orders = make_orders()
    
    # Run base orders
    for order in base_orders:
        if time.time() > deadline:
            break
        filled, placed = try_pack(order, deadline)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
    
    # Randomized restarts with diverse perturbation strategies
    rng = random.Random(12345)
    attempts = 0
    max_attempts = 800
    
    while time.time() < deadline and attempts < max_attempts:
        base = rng.choice(base_orders)
        order = list(base)
        
        # Choose perturbation strategy
        strategy = rng.random()
        if strategy < 0.3:
            # Random swaps
            for _ in range(rng.randint(1, 15)):
                i, j = rng.randrange(n), rng.randrange(n)
                order[i], order[j] = order[j], order[i]
        elif strategy < 0.5:
            # Segment reversal
            seg_len = rng.randint(2, min(15, n))
            start = rng.randrange(max(1, n - seg_len))
            order[start:start+seg_len] = reversed(order[start:start+seg_len])
        elif strategy < 0.7:
            # Move a random element to a different position
            i = rng.randrange(n)
            elem = order.pop(i)
            j = rng.randrange(len(order) + 1)
            order.insert(j, elem)
        else:
            # Insert a random box at a random position (simulates reordering)
            i = rng.randrange(n)
            elem = order.pop(i)
            j = rng.randrange(len(order) + 1)
            order.insert(j, elem)
            # Also swap two other elements
            if len(order) >= 2:
                k, l = rng.randrange(n), rng.randrange(n)
                if k != l:
                    order[k], order[l] = order[l], order[k]
        
        filled, placed = try_pack(order, deadline)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
        attempts += 1
        
        # If we've already filled the container completely, stop
        if best_filled >= container_vol:
            break
    
    # Construct output
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # Map placed boxes back to indices
    # placed is a list of (x, y, z, w, h, d, idx)
    for (px, py, pz, pw, ph, pd, idx) in best_placed:
        placements[idx] = (0, px, py, pz, pw, ph, pd)
        
    return placements
# EVOLVE-BLOCK-END
