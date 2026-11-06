"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []

    # Precompute unique orientations for each box
    # We store them as (w, h, d) tuples.
    box_orients = []
    for w, h, d in boxes:
        orients = set()
        # Generate all 6 permutations, but remove duplicates (e.g., if w==h)
        orients.add((w, h, d))
        orients.add((w, d, h))
        orients.add((h, w, d))
        orients.add((h, d, w))
        orients.add((d, w, h))
        orients.add((d, h, w))
        box_orients.append(list(orients))
    
    # Precompute volumes
    box_vols = [w*h*d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D

    # Grid parameters for spatial hashing
    # We need to choose a cell size. 
    # If cell size is too small, grid is huge. If too large, many false positives in collision check.
    # A reasonable heuristic is the median dimension of the boxes, or a fraction of the container.
    # Let's use a fixed step size for simplicity and robustness, or adapt it.
    # Let's try a step size that balances grid size and collision check cost.
    # If we use integer coordinates, we can use a grid of occupied cells.
    # However, boxes can be floating point? The problem implies integer or float.
    # Let's assume floats but snap to a grid for collision checking? 
    # No, standard 3D packing usually assumes exact coordinates.
    # Using a set of occupied cells in a grid is faster than O(N) overlap check.
    
    # Let's define a grid resolution. 
    # If dimensions are small integers, we can use a voxel grid.
    # If they are floats, we need a spatial hash.
    # Given "60-box instances", N is small. O(N^2) overlap check is 3600 ops per placement.
    # 60 boxes * 3600 = 216,000 ops per full pack.
    # With 110s budget, we can do many, many full packs.
    # Actually, the current program does O(N^2) per anchor. Anchors are O(N). So O(N^3) per pack.
    # 60^3 = 216,000. 
    # The bottleneck in the current program is likely the number of anchors and the O(N) fits check.
    # A simpler O(N) fits check with random restarts and best-fit might be faster and better.

    def get_orientations(idx):
        return box_orients[idx]

    def check_overlap(x, y, z, bw, bh, bd, placed):
        # placed is list of (x, y, z, w, h, d)
        for px, py, pz, pw, ph, pd in placed:
            # Check separation in any axis
            if x + bw <= px or px + pw <= x:
                continue
            if y + bh <= py or py + ph <= y:
                continue
            if z + bd <= pz or pz + pd <= z:
                continue
            return True
        return False

    def fits(x, y, z, bw, bh, bd, placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        if not check_overlap(x, y, z, bw, bh, bd, placed):
            return True
        return False

    def pack_once(order, rng):
        placed = []
        result = [None] * n
        current_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = get_orientations(idx)
            
            # Shuffle orientations to break symmetry
            rng.shuffle(orients)
            
            best_pos = None
            best_orient = None
            
            # Generate candidate positions
            # We use a "Best Fit" strategy: try to place as low, deep, left as possible.
            # But to be stochastic, we sample some anchors and also try random valid positions?
            # Random valid positions are hard to find efficiently without a grid.
            # Let's stick to anchors but add some randomness in selection.
            
            # Standard anchors: (0,0,0) and faces of existing boxes.
            anchors = [(0, 0, 0)]
            for px, py, pz, pw, ph, pd in placed:
                # Right
                anchors.append((px + pw, py, pz))
                # Front
                anchors.append((px, py + ph, pz))
                # Top
                anchors.append((px, py, pz + pd))
                # Corners (optional, might be too many)
                # Let's stick to faces for speed, maybe add some corners if N is small.
                # For 60 boxes, 3*60 = 180 anchors.
                # 180 anchors * 6 orients * O(N) check = 180*6*60 ~ 65k ops per box.
                # 60 boxes * 65k = 4M ops per pack.
                # 110s / 4M ops/s (python is slow, maybe 1M ops/s?) -> 110/4 = 27 packs?
                # That's low. We need to optimize.
                
            # Optimization: Use a grid for collision checking?
            # Or, simply reduce anchors.
            # Let's try a different approach: "First Fit Decreasing" with random restarts.
            
            # To speed up, we can limit the number of anchors checked.
            # Or, we can use a simpler heuristic: try (0,0,0) then scan? No.
            
            # Let's implement a faster check by maintaining a list of placed boxes and
            # checking only against nearby boxes? Hard in 3D without grid.
            
            # Let's try to improve the current anchor approach by:
            # 1. Sorting anchors by (z, y, x) to find "lowest" position quickly.
            # 2. Breaking early if found.
            
            valid_anchors = []
            for ax, ay, az in anchors:
                if 0 <= ax <= W and 0 <= ay <= H and 0 <= az <= D:
                    valid_anchors.append((ax, ay, az))
            
            # Sort to prioritize bottom-left-front
            valid_anchors.sort(key=lambda a: (a[2], a[1], a[0]))
            
            # Randomize the order of checking anchors slightly?
            # If we strictly sort, we always get the same result for same order.
            # To get diversity, we can shuffle the *placement order* of boxes,
            # and maybe shuffle the *orientation* choice.
            
            found = False
            # To introduce stochasticity in position, we can pick a random anchor from the top K lowest?
            # Or just rely on the random box order.
            
            # Let's try all anchors in sorted order, but if we find a fit, we take it.
            # This is deterministic for a given box order.
            # The randomness comes from the box order.
            
            for ax, ay, az in valid_anchors:
                for bw, bh, bd in orients:
                    if fits(ax, ay, az, bw, bh, bd, placed):
                        best_pos = (ax, ay, az)
                        best_orient = (bw, bh, bd)
                        found = True
                        break
                if found:
                    break
            
            if found:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                current_vol += bw * bh * bd
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                
        return result, current_vol

    # Strategies for ordering boxes
    # 1. Volume Desc
    # 2. Max Dim Desc
    # 3. Random
    # 4. Hybrid: Large boxes first, then randomize the rest?
    
    indexed_boxes = list(range(n))
    
    # Pre-sorted orders
    order_vol = sorted(indexed_boxes, key=lambda i: box_vols[i], reverse=True)
    order_max = sorted(indexed_boxes, key=lambda i: max(boxes[i]), reverse=True)
    
    deadline = time.time() + 105.0 # 5s margin for safety
    best_result = None
    best_vol = -1
    
    # First, try the deterministic best orders
    for order in [order_vol, order_max]:
        res, vol = pack_once(order, random.Random(42))
        if vol > best_vol:
            best_vol = vol
            best_result = res
    
    # Now, run random restarts
    # We need to be careful with time.
    # Each pack takes T seconds.
    # Estimate T by running one pack and timing it.
    
    start_time = time.time()
    # Run a sample pack to estimate time
    sample_order = list(indexed_boxes)
    random.shuffle(sample_order)
    t0 = time.time()
    res, vol = pack_once(sample_order, random.Random(42))
    t1 = time.time()
    pack_time = t1 - t0
    
    if vol > best_vol:
        best_vol = vol
        best_result = res
    
    # If pack is too slow, reduce iterations
    # If pack is very fast, increase iterations
    
    # Let's aim for ~50-100 iterations if possible
    # But adapt based on time remaining.
    
    while time.time() < deadline:
        # Create a random order
        # Weighted random? Large boxes are harder to place, so putting them first is generally good.
        # But sometimes a specific large box fits only if a smaller one is there? No, boxes don't depend on each other for fitting, only for space.
        # So "First Fit Decreasing" (by volume) is a strong baseline.
        # Random restarts with pure random order might be worse.
        # Let's try: 50% Vol Desc, 50% Random?
        # Or: Sort by volume, but swap adjacent elements with some probability?
        
        # Let's try a mix:
        # 1. Vol Desc
        # 2. Random
        # 3. Max Dim Desc
        # 4. "Greedy Random": Sort by volume, but with some noise?
        
        r = random.random()
        if r < 0.3:
            order = list(order_vol)
        elif r < 0.6:
            order = list(order_max)
        else:
            order = list(indexed_boxes)
            random.shuffle(order)
        
        # Add some noise to the order?
        # Swap a few elements
        for _ in range(5):
            i = random.randint(0, n-2)
            j = random.randint(i+1, n-1)
            order[i], order[j] = order[j], order[i]
            
        res, vol = pack_once(order, random.Random())
        
        if vol > best_vol:
            best_vol = vol
            best_result = res
            
        # Check time
        if time.time() > deadline:
            break
            
    # If best_result is still None (shouldn't happen), fallback
    if best_result is None:
        # Simple shelf packing fallback
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
        return placements

    return best_result
# EVOLVE-BLOCK-END
