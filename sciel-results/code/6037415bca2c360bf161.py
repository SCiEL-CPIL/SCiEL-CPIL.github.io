"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import bisect
import heapq

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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

    def try_placement(box_order, orientation_strategy='best_fit'):
        """
        Try to place boxes in the given order using free placement with candidate generation.
        Returns placements list and total volume.
        """
        placements = [None] * n
        placed_count = 0
        total_vol = 0

        placed_boxes = []  # list of (x, y, z, w, h, d)

        def can_place(x, y, z, w, h, d, exclude_idx=None):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Check overlap with all placed boxes using early exit
            for i, (px, py, pz, pw, ph, pd) in enumerate(placed_boxes):
                if exclude_idx is not None and i == exclude_idx:
                    continue
                # Quick rejection checks
                if x >= px + pw or px >= x + w:
                    continue
                if y >= py + ph or py >= y + h:
                    continue
                if z >= pz + pd or pz >= z + d:
                    continue
                return False
            return True

        def find_position(w, h, d, exclude_idx=None):
            """Find a valid position for a box with dimensions w, h, d."""
            # Generate candidate positions from extreme points
            candidates = [(0, 0, 0)]

            for i, (px, py, pz, pw, ph, pd) in enumerate(placed_boxes):
                if exclude_idx is not None and i == exclude_idx:
                    continue
                # Place to the right
                candidates.append((px + pw, py, pz))
                # Place in front
                candidates.append((px, py + ph, pz))
                # Place above
                candidates.append((px, py, pz + pd))
                # Combinations of two offsets
                candidates.append((px + pw, py + ph, pz))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
                candidates.append((px + pw, py + ph, pz + pd))

            # Deduplicate and filter
            seen = set()
            valid_candidates = []
            for (cx, cy, cz) in candidates:
                if (cx, cy, cz) in seen:
                    continue
                seen.add((cx, cy, cz))
                if can_place(cx, cy, cz, w, h, d, exclude_idx):
                    valid_candidates.append((cx, cy, cz))

            if not valid_candidates:
                return None

            # Score: prefer lower z, then y, then x (compactness)
            def pos_score(p):
                cx, cy, cz = p
                return (cz, cy, cx)

            valid_candidates.sort(key=pos_score)
            return valid_candidates[0]

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]

            if orientation_strategy == 'best_fit':
                # Try all orientations, pick the one that allows placement with lowest position
                best_orientation = None
                best_pos = None
                best_pos_score = None
                for (ow, oh, od) in box_orientations[idx]:
                    pos = find_position(ow, oh, od)
                    if pos is not None:
                        # Score the position
                        pos_score_val = (pos[2], pos[1], pos[0])
                        if best_pos_score is None or pos_score_val < best_pos_score:
                            best_pos_score = pos_score_val
                            best_pos = pos
                            best_orientation = (ow, oh, od)
                if best_pos is not None:
                    ow, oh, od = best_orientation
                    placed_boxes.append((best_pos[0], best_pos[1], best_pos[2], ow, oh, od))
                    placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], ow, oh, od)
                    placed_count += 1
                    total_vol += ow * oh * od
            else:
                # Use first orientation
                ow, oh, od = box_orientations[idx][0]
                pos = find_position(ow, oh, od)
                if pos is not None:
                    placed_boxes.append((pos[0], pos[1], pos[2], ow, oh, od))
                    placements[idx] = (0, pos[0], pos[1], pos[2], ow, oh, od)
                    placed_count += 1
                    total_vol += ow * oh * od

        # Fill in unplaced boxes
        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    def local_search(placements, placed_indices, max_iterations=50):
        """
        Local search: try to move placed boxes to better positions to improve packing.
        Returns improved placements and total volume.
        """
        if not placed_indices:
            return placements, 0

        # Rebuild placed_boxes from placements
        placed_boxes = []
        idx_map = {}
        for i in placed_indices:
            p = placements[i]
            if p[0] == 0:
                x, y, z, w, h, d = p[1], p[2], p[3], p[4], p[5], p[6]
                placed_boxes.append((x, y, z, w, h, d))
                idx_map[len(placed_boxes) - 1] = i

        total_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)

        def can_place(x, y, z, w, h, d, exclude_pb_idx=None):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for i, (px, py, pz, pw, ph, pd) in enumerate(placed_boxes):
                if exclude_pb_idx is not None and i == exclude_pb_idx:
                    continue
                if x >= px + pw or px >= x + w:
                    continue
                if y >= py + ph or py >= y + h:
                    continue
                if z >= pz + pd or pz >= z + d:
                    continue
                return False
            return True

        def find_position(w, h, d, exclude_pb_idx=None):
            candidates = [(0, 0, 0)]
            for i, (px, py, pz, pw, ph, pd) in enumerate(placed_boxes):
                if exclude_pb_idx is not None and i == exclude_pb_idx:
                    continue
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px, py, pz + pd))
                candidates.append((px + pw, py + ph, pz))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
                candidates.append((px + pw, py + ph, pz + pd))

            seen = set()
            valid_candidates = []
            for (cx, cy, cz) in candidates:
                if (cx, cy, cz) in seen:
                    continue
                seen.add((cx, cy, cz))
                if can_place(cx, cy, cz, w, h, d, exclude_pb_idx):
                    valid_candidates.append((cx, cy, cz))

            if not valid_candidates:
                return None

            def pos_score(p):
                cx, cy, cz = p
                return (cz, cy, cx)

            valid_candidates.sort(key=pos_score)
            return valid_candidates[0]

        improved = False
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            improved_this_round = False
            for pb_idx in range(len(placed_boxes)):
                if time.time() > deadline - 5:
                    break
                orig_idx = idx_map[pb_idx]
                x, y, z, w, h, d = placed_boxes[pb_idx]
                # Try to find a better position
                new_pos = find_position(w, h, d, pb_idx)
                if new_pos is not None:
                    nx, ny, nz = new_pos
                    # Only move if it improves the score (lower position)
                    if (nz, ny, nx) < (z, y, x):
                        placed_boxes[pb_idx] = (nx, ny, nz, w, h, d)
                        placements[orig_idx] = (0, nx, ny, nz, w, h, d)
                        improved_this_round = True
            if not improved_this_round:
                break
            improved = improved_this_round

        return placements, total_vol

    def shelf_packing(box_order, use_rotation=True):
        """
        Shelf-based packing: boxes are placed on horizontal shelves.
        """
        placements = [None] * n
        placed_count = 0
        total_vol = 0

        shelves = []  # list of shelf dicts

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]

            if not use_rotation:
                orientations = [(w_orig, h_orig, d_orig)]
            else:
                orientations = box_orientations[idx]

            best_shelf_idx = None
            best_pos = None
            best_orientation = None
            best_score = None

            for (ow, oh, od) in orientations:
                # Try to place on an existing shelf
                for s_idx, s in enumerate(shelves):
                    # Shelf height must accommodate this box
                    if oh <= s['h'] and s['x'] + ow <= W and od <= D:
                        pos = (s['x'], 0, s['z'])
                        # Score: prefer lower z, then lower x
                        score = (s['z'], s['x'])
                        if best_score is None or score < best_score:
                            best_score = score
                            best_shelf_idx = s_idx
                            best_pos = pos
                            best_orientation = (ow, oh, od)

                # Try to create a new shelf
                if oh <= H:
                    # Find the minimum z where a new shelf of height oh can fit
                    min_z = 0
                    for s in shelves:
                        if s['z'] + s['h'] > min_z:
                            min_z = s['z'] + s['h']

                    if min_z + oh <= H:
                        pos = (0, 0, min_z)
                        score = (min_z, 0)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_shelf_idx = -1  # signal to create new shelf
                            best_pos = pos
                            best_orientation = (ow, oh, od)

            if best_shelf_idx is not None and best_pos is not None:
                ow, oh, od = best_orientation
                if best_shelf_idx >= 0:
                    # Place on existing shelf
                    shelf = shelves[best_shelf_idx]
                    shelf['x'] += ow
                else:
                    # Create new shelf
                    new_shelf = {'z': best_pos[2], 'h': oh, 'x': ow, 'boxes': []}
                    shelves.append(new_shelf)

                placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2], ow, oh, od)
                placed_count += 1
                total_vol += ow * oh * od

        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    def layer_packing(box_order, use_rotation=True):
        """
        Layer-based packing similar to the inspiration.
        """
        placements = [None] * n
        placed_count = 0
        total_vol = 0

        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0

        for idx in box_order:
            if time.time() > deadline:
                break

            w_orig, h_orig, d_orig = boxes[idx]

            if not use_rotation:
                orientations = [(w_orig, h_orig, d_orig)]
            else:
                orientations = box_orientations[idx]

            best_orientation = None
            best_x, best_y, best_z = None, None, None
            best_row_h, best_layer_d = None, None
            best_score = None

            for (ow, oh, od) in orientations:
                # Simulate placement
                sx, sy, sz = x, y, z
                srow_h, slayer_d = row_h, layer_d

                if sx + ow > W:
                    sx, sy, srow_h = 0, sy + srow_h, 0
                if sy + oh > H:
                    sy, sz, slayer_d, sx, srow_h = 0, sz + slayer_d, 0, 0, 0
                if sz + od > D:
                    continue  # Doesn't fit

                # This orientation fits. Score it.
                score = (sz, sy, sx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_orientation = (ow, oh, od)
                    best_x, best_y, best_z = sx, sy, sz
                    best_row_h, best_layer_d = srow_h, slayer_d

            if best_orientation is not None:
                ow, oh, od = best_orientation
                placements[idx] = (0, best_x, best_y, best_z, ow, oh, od)
                placed_count += 1
                total_vol += ow * oh * od

                # Update state
                x = best_x + ow
                y = best_y
                z = best_z
                row_h = max(best_row_h, oh)
                layer_d = max(best_layer_d, od)

        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, total_vol

    # Define sorting strategies
    strategies = []
    
    # 1. Volume Descending
    strategies.append(sorted(range(n), key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2])))
    # 2. Max Dimension Descending
    strategies.append(sorted(range(n), key=lambda i: -max(boxes[i])))
    # 3. Min Dimension Ascending
    strategies.append(sorted(range(n), key=lambda i: min(boxes[i])))
    # 4. Max Dimension Ascending
    strategies.append(sorted(range(n), key=lambda i: max(boxes[i])))
    # 5. Volume Ascending
    strategies.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2]))
    # 6. Surface Area Descending
    strategies.append(sorted(range(n), key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]))))
    # 7. Aspect Ratio (Max/Min) Descending
    strategies.append(sorted(range(n), key=lambda i: -(max(boxes[i]) / max(1, min(boxes[i])))))
    # 8. Aspect Ratio (Max/Min) Ascending
    strategies.append(sorted(range(n), key=lambda i: (max(boxes[i]) / max(1, min(boxes[i])))))
    # 9. Second-largest dimension descending
    strategies.append(sorted(range(n), key=lambda i: -sorted(boxes[i], reverse=True)[1]))
    # 10. Perimeter descending (2*(w+h+d))
    strategies.append(sorted(range(n), key=lambda i: -(2*(boxes[i][0]+boxes[i][1]+boxes[i][2]))))

    best_placements = None
    best_vol = 0

    # Run deterministic strategies with Free Placement
    for order in strategies:
        if time.time() > deadline:
            break
        result = try_placement(order, 'best_fit')
        if result:
            _, vol = result
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]

    # Run Shelf Packing with top strategies
    for order in strategies[:4]:
        if time.time() > deadline:
            break
        result = shelf_packing(order, True)
        if result:
            _, vol = result
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]

    # Run Layer Packing with top strategies
    for order in strategies[:4]:
        if time.time() > deadline:
            break
        result = layer_packing(order, True)
        if result:
            _, vol = result
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]

    # Randomized Restart with Free Placement
    # Since N=60 is small, we can afford many random restarts
    rng = random.Random(42)
    random_trials = 0
    max_random_trials = 150  # Reduced to leave time for local search
    
    while time.time() < deadline - 20.0:
        # Mix of fully random and partially sorted orders
        if random_trials % 3 == 0:
            order = list(range(n))
            rng.shuffle(order)
        elif random_trials % 3 == 1:
            # Start with large boxes, then randomize the rest
            order = sorted(range(n), key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2]))
            # Randomize last 2/3
            split = n // 3
            tail = order[split:]
            rng.shuffle(tail)
            order = order[:split] + tail
        else:
            # Weighted random: larger boxes more likely to come first
            order = list(range(n))
            weights = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
            total_w = sum(weights)
            # Simple weighted shuffle
            for i in range(n - 1, 0, -1):
                r = rng.random() * sum(weights[:i+1])
                acc = 0
                for j in range(i + 1):
                    acc += weights[j]
                    if acc >= r:
                        order[i], order[j] = order[j], order[i]
                        weights[i], weights[j] = weights[j], weights[i]
                        break
        result = try_placement(order, 'best_fit')
        if result:
            _, vol = result
            if vol > best_vol:
                best_vol = vol
                best_placements = result[0]
        random_trials += 1
        if random_trials >= max_random_trials:
            break

    # Apply local search to the best solution found
    if best_placements is not None and time.time() < deadline - 5.0:
        placed_indices = [i for i in range(n) if best_placements[i][0] == 0]
        best_placements, best_vol = local_search(best_placements, placed_indices, max_iterations=30)

    if best_placements is None:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    return best_placements

# EVOLVE-BLOCK-END
