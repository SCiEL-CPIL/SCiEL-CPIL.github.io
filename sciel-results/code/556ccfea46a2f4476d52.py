"""3D Container Loading - Skyline-based approach with multiple orientations and ILS."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s kill
    
    container_vol = W * H * D
    
    # Precompute unique orientations for each box
    orient = []
    for w, h, d in boxes:
        opts = set()
        for p in itertools.permutations((w, h, d)):
            opts.add(p)
        orient.append(list(opts))
    
    box_vol = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    def try_pack(order, time_limit):
        """
        Fast 3D pack using a skyline/floor-map representation.
        We maintain a grid of minimum heights (z-coordinates) for the X-Y plane.
        To handle Y-dimension effectively in a 2D grid, we can't just use a single Z-height 
        per (x,y) if boxes have different heights in Y. 
        However, the standard "Skyline" or "Extremal Point" approach in 2D bin packing 
        uses a list of (x, width, height) segments. In 3D, a common heuristic is to 
        maintain a list of "free space" rectangles or use a 2D grid if dimensions are small.
        
        Given the constraint of arbitrary dimensions, a full 3D grid is too memory intensive.
        Instead, we use a hybrid: 
        1. Maintain a list of placed boxes.
        2. Use a "skyline" in the Z-direction for each (x, y) cell? No, that's still 3D.
        
        Let's stick to the "Anchor" approach but optimize it, OR use a simplified Skyline 
        that works on the X-Y plane assuming we fill layers? 
        The prompt suggests: "maintains a 2D grid of minimum heights for each cell".
        This implies we discretize X and Y? If W, H are large, this is bad.
        
        Alternative interpretation of "Skyline" in this context:
        Maintain a list of "free rectangles" in the X-Y plane at various Z levels?
        
        Let's implement a robust "Extremal Points" (Anchor) approach first, as it's 
        dimension-agnostic and O(1) to update, but O(N) to check. 
        However, the prompt specifically asks for a "skyline" or "floor-map" representation 
        that allows O(1) overlap checks. This usually implies a grid.
        
        If we assume the container dimensions are not huge (e.g., < 1000), we can use a 3D grid 
        or a 2D grid of heights. Let's try a 2D grid of heights (Z) for the X-Y plane.
        This only works if we treat the container as a stack of layers? No, boxes can have different heights.
        
        Actually, a "Floor Map" typically stores the height of the floor at each (x, y) point.
        If we place a box of height h at (x, y), we update the floor map for all (x', y') in the box's footprint 
        to be at least current_floor + h? No, that's for 2D packing into a 3D volume where 
        we want to minimize Z.
        
        Let's try this: 
        Grid `floor` of size W x H (or scaled down if too large).
        `floor[x][y]` is the minimum Z coordinate where a new box can be placed at (x, y).
        To place a box of size (w, h, d) at (x, y, z):
        - z must be >= max(floor[x:x+w][y:y+h])
        - Check if z + d <= D
        - Check overlap with other boxes? 
          If we only store max Z, we lose information about "holes" underneath. 
          Example: Box A at (0,0,0) size 1x1x1. Box B at (0,0,1) size 1x1x1.
          Floor at (0,0) becomes 2.
          Box C at (0,0,0) size 1x1x1. 
          Floor check: z >= 2. So it places at z=2. But it could have fit at z=0 if we knew A was there? 
          Wait, A is at (0,0,0). So (0,0,0) is occupied. The floor map correctly says you can't place at z=0 
          because the floor is "raised" to 2? No, the floor map usually represents the *bottom* of the free space.
          If A is at (0,0,0) to (1,1,1), the free space starts at z=1. So floor[0][0] = 1.
          If we place B at (0,0,1) to (1,1,2), the free space at (0,0) starts at z=2. So floor[0][0] = 2.
          This works for "stacking". But what if we want to place a box next to A, not on top?
          Box C at (1,0,0). 
          Floor[1][0] is initially 0. 
          Max floor in footprint (1,0) to (2,1) is 0. So z=0 is valid.
          Does it overlap A? A is x:0-1, y:0-1. C is x:1-2, y:0-1. No overlap.
          This seems to work for non-overlapping boxes if we update the floor map correctly.
          
        Update rule:
        When placing a box at (x, y, z) with size (w, h, d):
        For all x' in [x, x+w), y' in [y, y+h):
            floor[x'][y'] = max(floor[x'][y'], z + d)
            
        This effectively "raises the floor" for that footprint.
        This is a valid heuristic for 3D bin packing (often called the "Skyline" or "Height Map" approach).
        It guarantees no overlap because any new box must start at or above the current floor height for all cells in its footprint.
        
        Complexity: O(W*H) to initialize, O(w*h) to place a box.
        If W, H are large (e.g., 1000x1000), this might be slow or memory heavy. 
        But 60 boxes is small. If W, H are reasonable (e.g., < 200), this is very fast.
        If W, H are huge (e.g., 10000), we might need to scale or use anchors.
        
        Let's check typical container sizes. If they are small integers, this is perfect.
        If they are large, we can scale down the grid or fall back to anchors.
        For now, let's assume grid is feasible. We'll cap the grid size to prevent memory explosion.
        """
        # Determine if we should use Grid or Anchor
        # If W*H is too large, use Anchor.
        if W * H > 100000:
            return try_pack_anchors(order, time_limit)
            
        # Initialize floor map
        # Using a list of lists for speed
        floor = [[0] * H for _ in range(W)]
        
        placed = []  # (x, y, z, w, h, d, box_idx)
        filled = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_key = None
            
            for (ow, oh, od) in orient[idx]:
                if ow > W or oh > H or od > D:
                    continue
                
                # Find best (x, y) position
                # We want to minimize z, then y, then x (or similar heuristic)
                # To do this efficiently, we can iterate x, y and find the min z.
                # But iterating all x, y is O(W*H).
                # We can optimize by only checking positions where the floor might be low?
                # For simplicity, let's just scan. If W*H is <= 100k, and we have 60 boxes,
                # 60 * 100k * 6 orientations = 36M operations. Might be too slow in Python.
                
                # Optimization: Only check x, y where floor[x][y] is minimal?
                # Or just do a full scan but break early if we find a good spot?
                
                # Let's try a heuristic scan:
                # Iterate y from 0 to H-oh
                # Iterate x from 0 to W-ow
                # z = max(floor[x:x+ow][y:y+oh])
                # If z + od <= D, it's a valid spot.
                # Key: (z, y, x)
                
                # To speed up, we can keep track of the minimum z seen so far for this orientation?
                # No, z depends on x, y.
                
                # Let's just do the scan. If it's too slow, we'll switch to anchors.
                # We can add a time check inside the loop.
                
                found_best = False
                for y in range(0, H - oh + 1):
                    if time.time() > time_limit:
                        return filled, placed
                    for x in range(0, W - ow + 1):
                        # Calculate max z in footprint
                        max_z = 0
                        # Optimization: if we know the max z for this row segment... 
                        # For now, simple loop
                        for dy in range(oh):
                            row = floor[x:x+ow]  # This is a slice, creates a new list. Bad.
                            # Better:
                            # for dx in range(ow):
                            #     val = floor[x+dx][y+dy]
                            #     if val > max_z: max_z = val
                            
                            # Let's do it manually to avoid slice overhead
                            pass
                        
                        # Re-implement inner loop properly
                        max_z = 0
                        for dy in range(oh):
                            yy = y + dy
                            for dx in range(ow):
                                xx = x + dx
                                val = floor[xx][yy]
                                if val > max_z:
                                    max_z = val
                                    
                        if max_z + od > D:
                            continue
                            
                        key = (max_z, y, x)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (x, y, max_z, ow, oh, od)
                            
                        # If we found (0,0,0) orientation, it's the best possible for this orient?
                        # Not necessarily, another orient might have a smaller z? No, z is determined by floor.
                        # But if z=0, it's the lowest possible.
                        if best_key == (0, 0, 0):
                            found_best = True
                            break
                    if found_best:
                        break
                
                if best_pos is not None and best_key == (0, 0, 0):
                    break # Can't do better than z=0, y=0, x=0
                
            if best_pos is not None:
                bx, by, bz, bw, bh, bd = best_pos
                placed.append((bx, by, bz, bw, bh, bd, idx))
                filled += bw * bh * bd
                
                # Update floor map
                for dy in range(bh):
                    yy = by + dy
                    for dx in range(bw):
                        xx = bx + dx
                        new_h = bz + bd
                        if new_h > floor[xx][yy]:
                            floor[xx][yy] = new_h
                
                if time.time() > time_limit:
                    break
        
        return filled, placed

    def try_pack_anchors(order, time_limit):
        """Fallback anchor-based packing for large containers."""
        placed = []
        filled = 0
        anchors = [(0, 0, 0)]
        anchor_set = {(0, 0, 0)}
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_key = None
            
            for (ow, oh, od) in orient[idx]:
                if ow > W or oh > H or od > D:
                    continue
                
                for (ax, ay, az) in anchors:
                    if ax + ow > W or ay + oh > H or az + od > D:
                        continue
                    
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
                        
                if time.time() > time_limit:
                    break
        
        return filled, placed

    def make_orders():
        orders = []
        orders.append(sorted(range(n), key=lambda i: -box_vol[i]))
        orders.append(sorted(range(n), key=lambda i: box_vol[i]))
        orders.append(sorted(range(n), key=lambda i: -min(boxes[i])))
        orders.append(sorted(range(n), key=lambda i: -max(boxes[i])))
        def surf(i):
            a, b, c = boxes[i]
            return 2 * (a * b + a * c + b * c)
        orders.append(sorted(range(n), key=lambda i: -surf(i)))
        def compact(i):
            a, b, c = boxes[i]
            m = max(a, b, c)
            return (a * b * c) / m if m else 0
        orders.append(sorted(range(n), key=lambda i: -compact(i)))
        return orders

    best_filled = -1
    best_placed = []
    best_order = None
    
    base_orders = make_orders()
    
    # Run base orders
    for order in base_orders:
        if time.time() > deadline:
            break
        filled, placed = try_pack(order, deadline)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
            best_order = list(order)
            
    # ILS
    rng = random.Random(12345)
    attempts = 0
    max_attempts = 5000
    
    while time.time() < deadline and attempts < max_attempts:
        if best_order is None:
            order = list(rng.choice(base_orders))
        else:
            order = list(best_order)
            
        strategy = rng.random()
        if strategy < 0.3:
            for _ in range(rng.randint(1, 5)):
                i, j = rng.randrange(n), rng.randrange(n)
                order[i], order[j] = order[j], order[i]
        elif strategy < 0.6:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i > j: i, j = j, i
            order[i:j+1] = reversed(order[i:j+1])
        else:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j: continue
            # Move i to j
            item = order.pop(i)
            order.insert(j, item)
            
        filled, placed = try_pack(order, deadline)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
            best_order = list(order)
        attempts += 1

    # Build final placements
    final_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    for (bx, by, bz, bw, bh, bd, idx) in best_placed:
        final_placements[idx] = (0, bx, by, bz, bw, bh, bd)
        
    return final_placements
# EVOLVE-BLOCK-END
