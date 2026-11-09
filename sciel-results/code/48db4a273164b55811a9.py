"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import bisect

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    n = len(boxes)

    # Precompute all valid orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        orientations = set()
        for perm in itertools.permutations((w, h, d)):
            orientations.add(perm)
        box_orientations.append(list(orientations))

    # Sort strategies
    strategies = [
        lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2]),  # volume desc
        lambda i: -max(boxes[i]),  # max dim desc
        lambda i: min(boxes[i]),  # min dim asc
        lambda i: boxes[i][0] * boxes[i][1],  # area desc (tie: volume)
        lambda i: -min(boxes[i]),  # min dim desc
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],  # sum asc
    ]

    def try_placement(box_order, seed=None, scoring='compact'):
        """
        Place boxes using corner-based candidate generation.
        scoring: 'compact' (min z,y,x), 'gap_fill' (maximize space utilization)
        """
        placements = [None] * n
        total_vol = 0

        placed_boxes = []  # list of (x, y, z, w, h, d)
        
        # For faster overlap checking, maintain sorted structures
        # But for 60 boxes, simple O(n) check with early exit is fast enough
        
        def can_place_fast(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Early exit coordinate checks
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                if x < px + pw and x + w > px:
                    if y < py + ph and y + h > py:
                        if z < pz + pd and z + d > pz:
                            return False
            return True

        def generate_candidates(w, h, d):
            """Generate candidate positions based on placed box corners."""
            candidates = set()
            candidates.add((0, 0, 0))
            
            # For each placed box, add positions adjacent to it
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # Right face
                candidates.add((px + pw, py, pz))
                # Left face (if not at origin)
                if px > 0:
                    candidates.add((px - w, py, pz))
                # Front face
                candidates.add((px, py + ph, pz))
                if py > 0:
                    candidates.add((px, py - h, pz))
                # Top face
                candidates.add((px, py, pz + pd))
                if pz > 0:
                    candidates.add((px, py, pz - d))
                
                # Corner combinations
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
                
                # Gap-filling: try to fill gaps next to boxes
                if px > 0:
                    candidates.add((px - w, py + ph, pz))
                    candidates.add((px - w, py, pz + pd))
                if py > 0:
                    candidates.add((px + pw, py - h, pz))
                    candidates.add((px, py - h, pz + pd))
                if pz > 0:
                    candidates.add((px + pw, py, pz - d))
                    candidates.add((px, py + ph, pz - d))

            # Filter valid candidates
            valid = []
            for (cx, cy, cz) in candidates:
                if cx < 0 or cy < 0 or cz < 0:
                    continue
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                if can_place_fast(cx, cy, cz, w, h, d):
                    valid.append((cx, cy, cz))
            
            return valid

        def score_position(pos, w, h, d, scoring_mode):
            """Score a position for preference."""
            x, y, z = pos
            if scoring_mode == 'compact':
                # Prefer lower z, then lower y, then lower x
                return (z, y, x)
            elif scoring_mode == 'gap_fill':
                # Prefer positions that maximize remaining space in x, y, z
                # Lower x,y,z is better, but also consider how much space is left
                remaining_x = W - (x + w)
                remaining_y = H - (y + h)
                remaining_z = D - (z + d)
                # Want to minimize wasted space, so prefer positions where box fits snugly
                # Actually, for gap filling, we want to place in gaps, so prefer
                # positions adjacent to existing boxes
                # Simple heuristic: prefer lower z, then lower y, then lower x
                # But with a slight preference for positions that don't leave large gaps
                return (z, y, x, -remaining_x, -remaining_y, -remaining_z)
            else:
                return (z, y, x)

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]
            best_orientation = None
            best_pos = None
            best_score = None

            for (ow, oh, od) in box_orientations[idx]:
                candidates = generate_candidates(ow, oh, od)
                if not candidates:
                    continue
                
                for pos in candidates:
                    pos_score = score_position(pos, ow, oh, od, scoring)
                    if best_score is None or pos_score < best_score:
                        best_score = pos_score
                        best_pos = pos
                        best_orientation = (ow, oh, od)
                
                if best_pos is not None:
                    break  # Found a valid orientation, use the best position

            if best_pos is not None:
                ow, oh, od = best_orientation
                placed_boxes.append((best_pos[0], best_pos[1], best_pos[2], ow, oh, od))
                placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], ow, oh, od)
                total_vol += ow * oh * od

        # Fill in unplaced boxes
        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    def shelf_packing(box_order, use_rotation=True, scoring='compact'):
        """Shelf-based packing with rotation."""
        placements = [None] * n
        total_vol = 0

        # Shelves: list of (z, height, current_x)
        shelves = []

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]
            orientations = box_orientations[idx] if use_rotation else [(w_orig, h_orig, d_orig)]

            best_shelf_idx = None
            best_pos = None
            best_orientation = None
            best_score = None

            for (ow, oh, od) in orientations:
                # Try existing shelves
                for si, (sz, sh, sx) in enumerate(shelves):
                    if sh == oh and sx + ow <= W and od <= D:
                        pos = (sx, 0, sz)
                        pos_score = (sz, sx)
                        if best_score is None or pos_score < best_score:
                            best_score = pos_score
                            best_shelf_idx = si
                            best_pos = pos
                            best_orientation = (ow, oh, od)
                
                # Try new shelf
                if oh <= H:
                    min_z = 0
                    for (sz, sh, sx) in shelves:
                        if sz + sh > min_z:
                            min_z = sz + sh
                    if min_z + oh <= H:
                        pos = (0, 0, min_z)
                        pos_score = (min_z, 0)
                        if best_score is None or pos_score < best_score:
                            best_score = pos_score
                            best_shelf_idx = len(shelves)  # new shelf
                            best_pos = pos
                            best_orientation = (ow, oh, od)

            if best_pos is not None and best_orientation is not None:
                ow, oh, od = best_orientation
                if best_shelf_idx < len(shelves):
                    # Update existing shelf
                    sz, sh, sx = shelves[best_shelf_idx]
                    shelves[best_shelf_idx] = (sz, sh, sx + ow)
                else:
                    # Create new shelf
                    shelves.append((best_pos[2], oh, ow))
                
                placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], ow, oh, od)
                total_vol += ow * oh * od

        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    # Try multiple strategies and pick the best
    best_placements = None
    best_vol = 0

    # Strategy 1: Volume desc, compact scoring
    box_order1 = sorted(range(n), key=strategies[0])
    result1 = try_placement(box_order1, scoring='compact')
    if result1:
        best_placements, best_vol = result1

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 2: Max dim desc, compact scoring
    box_order2 = sorted(range(n), key=strategies[1])
    result2 = try_placement(box_order2, scoring='compact')
    if result2:
        _, vol2 = result2
        if vol2 > best_vol:
            best_vol = vol2
            best_placements = result2[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 3: Min dim asc, compact scoring
    box_order3 = sorted(range(n), key=strategies[2])
    result3 = try_placement(box_order3, scoring='compact')
    if result3:
        _, vol3 = result3
        if vol3 > best_vol:
            best_vol = vol3
            best_placements = result3[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 4: Area desc, gap_fill scoring
    box_order4 = sorted(range(n), key=strategies[3])
    result4 = try_placement(box_order4, scoring='gap_fill')
    if result4:
        _, vol4 = result4
        if vol4 > best_vol:
            best_vol = vol4
            best_placements = result4[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 5: Shelf packing with volume desc
    result5 = shelf_packing(box_order1, True, 'compact')
    if result5:
        _, vol5 = result5
        if vol5 > best_vol:
            best_vol = vol5
            best_placements = result5[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Strategy 6: Shelf packing with max dim desc
    result6 = shelf_packing(box_order2, True, 'compact')
    if result6:
        _, vol6 = result6
        if vol6 > best_vol:
            best_vol = vol6
            best_placements = result6[0]

    if time.time() > deadline:
        if best_placements is None:
            return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        return best_placements

    # Random restarts with different orderings
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    max_restarts = int(min(200, remaining_time / 0.1))  # ~100ms per restart
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering with some structure
        order = list(range(n))
        # Mix of random and sorted
        if restart % 3 == 0:
            # Volume-weighted random
            weights = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
            for i in range(n - 1, 0, -1):
                total_w = sum(weights[:i + 1])
                r = rng.random() * total_w
                cumulative = 0
                for j in range(i + 1):
                    cumulative += weights[j]
                    if cumulative >= r:
                        order[i], order[j] = order[j], order[i]
                        weights[i], weights[j] = weights[j], weights[i]
                        break
        elif restart % 3 == 1:
            # Random shuffle
            rng.shuffle(order)
        else:
            # Sorted by random key
            rng.shuffle(order)
            order.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2] * rng.random())
        
        # Choose scoring mode
        scoring = 'compact' if restart % 2 == 0 else 'gap_fill'
        
        result = try_placement(order, scoring=scoring)
        if result:
            _, vol = result
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]

    if best_placements is None:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    return best_placements

# EVOLVE-BLOCK-END
