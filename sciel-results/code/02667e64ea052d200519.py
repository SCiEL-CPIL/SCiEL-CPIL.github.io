"""3D Container Loading - Anchor-based Best-Fit with Iterated Local Search."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we aim to finish by 140s to be safe
    deadline = start_time + 135.0
    
    container_vol = W * H * D
    
    # Precompute unique orientations for each box
    # For each box, get all unique permutations of (w, h, d)
    orient = []
    for w, h, d in boxes:
        opts = set()
        for p in itertools.permutations((w, h, d)):
            if p[0] <= W and p[1] <= H and p[2] <= D:
                opts.add(p)
        # If no orientation fits, the box cannot be placed
        if not opts:
            orient.append([])
        else:
            orient.append(list(opts))
    
    box_vol = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    def try_pack(order, time_limit):
        """
        Packs boxes in the given order using anchor-based best-fit.
        Returns (filled_volume, placements_dict)
        placements_dict maps box_index to (x, y, z, w, h, d) or None if not placed.
        """
        placed_boxes = []  # List of (x, y, z, w, h, d, idx)
        filled = 0
        # Anchors are (x, y, z). Start with origin.
        anchors = [(0, 0, 0)]
        anchor_set = {(0, 0, 0)}
        
        # To speed up overlap checks, we keep placed boxes in a list.
        # For N=60, O(N) overlap check is fine.
        
        for idx in order:
            if time.time() > time_limit:
                break
                
            w_orig, h_orig, d_orig = boxes[idx]
            best_pos = None
            best_key = None
            best_orient = None
            
            # Try all orientations for this box
            for (ow, oh, od) in orient[idx]:
                if not orient[idx]:
                    continue
                    
                # Iterate through current anchors
                # Optimization: We can break early if we find a position at (0,0,0)
                # but anchors are not sorted. We just track the best key.
                for (ax, ay, az) in anchors:
                    if ax + ow > W or ay + oh > H or az + od > D:
                        continue
                    
                    # Check overlap with all placed boxes
                    ok = True
                    for (px, py, pz, pw, ph, pd, pidx) in placed_boxes:
                        # Disjoint if separated in any dimension
                        if not (ax + ow <= px or px + pw <= ax or
                                ay + oh <= py or py + ph <= ay or
                                az + od <= pz or pz + pd <= az):
                            ok = False
                            break
                    
                    if ok:
                        # Key: Lowest Z, then Lowest Y, then Lowest X
                        key = (az, ay, ax)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (ax, ay, az)
                            best_orient = (ow, oh, od)
                        
                        # If we found the absolute best position (0,0,0), we can't do better
                        if best_key == (0, 0, 0):
                            break
                
                # If best position is origin, no other orientation can be better
                if best_key == (0, 0, 0):
                    break
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed_boxes.append((bx, by, bz, bw, bh, bd, idx))
                filled += bw * bh * bd
                
                # Generate new anchors from the placed box
                # New anchors are the corners of the placed box that are on the "outer" boundaries
                # or adjacent to existing space.
                # Standard set of new anchors to consider:
                new_anchors = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                    (bx + bw, by + bh, bz + bd)
                ]
                
                for na in new_anchors:
                    if na not in anchor_set:
                        # Optional: Prune anchors that are "blocked" by other boxes?
                        # For simplicity and speed, we just add them.
                        # We can also prune if an anchor is inside an existing box.
                        # Check if anchor is inside any placed box
                        inside = False
                        for (px, py, pz, pw, ph, pd, pidx) in placed_boxes:
                            if px <= na[0] < px + pw and py <= na[1] < py + ph and pz <= na[2] < pz + pd:
                                inside = True
                                break
                        if not inside:
                            anchor_set.add(na)
                            anchors.append(na)
            # else: box not placed, skip
            
        # Construct result
        placements = {}
        for (px, py, pz, pw, ph, pd, pidx) in placed_boxes:
            placements[pidx] = (px, py, pz, pw, ph, pd)
            
        return filled, placements

    def make_orders():
        orders = []
        # 1. Descending Volume
        orders.append(sorted(range(n), key=lambda i: -box_vol[i]))
        # 2. Descending Max Dimension
        orders.append(sorted(range(n), key=lambda i: -max(boxes[i])))
        # 3. Descending Surface Area
        def surf(i):
            a, b, c = boxes[i]
            return 2 * (a * b + a * c + b * c)
        orders.append(sorted(range(n), key=lambda i: -surf(i)))
        # 4. Descending Min Dimension
        orders.append(sorted(range(n), key=lambda i: -min(boxes[i])))
        # 5. Ascending Volume (sometimes helps fit small gaps later, though less common)
        orders.append(sorted(range(n), key=lambda i: box_vol[i]))
        return orders

    best_filled = -1
    best_placed = {}
    best_order = None
    
    base_orders = make_orders()
    
    # Phase 1: Deterministic Heuristics
    for order in base_orders:
        if time.time() > deadline:
            break
        filled, placed = try_pack(order, deadline)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
            best_order = list(order)
    
    # Phase 2: Iterated Local Search (ILS)
    rng = random.Random(42)
    attempts = 0
    max_attempts = 5000 # High cap, but time-limited
    
    while time.time() < deadline and attempts < max_attempts:
        if best_order is None:
            order = list(rng.choice(base_orders))
        else:
            order = list(best_order)
        
        # Perturbation Strategy
        strategy = rng.random()
        if strategy < 0.4:
            # Random Swap
            i, j = rng.randrange(n), rng.randrange(n)
            order[i], order[j] = order[j], order[i]
        elif strategy < 0.7:
            # Segment Reversal
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i > j: i, j = j, i
            order[i:j+1] = reversed(order[i:j+1])
        else:
            # Random Move
            i = rng.randrange(n)
            j = rng.randrange(n)
            item = order.pop(i)
            order.insert(j, item)
        
        filled, placed = try_pack(order, deadline)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
            best_order = list(order)
            
        attempts += 1
        
        # Adaptive cooling: if we've had many attempts without improvement, maybe restart from a random base order?
        # For now, simple ILS is sufficient.

    # Construct final output
    results = []
    for i in range(n):
        if i in best_placed:
            x, y, z, w, h, d = best_placed[i]
            results.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            results.append((-1, 0, 0, 0, w, h, d))
            
    return results
# EVOLVE-BLOCK-END
