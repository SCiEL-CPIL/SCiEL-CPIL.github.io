import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        perms = set(itertools.permutations([w, h, d]))
        return list(perms)

    box_orientations = [get_orientations(w, h, d) for (w, h, d) in boxes]

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True

    def check_overlap_with_placed(cx, cy, cz, ow, oh, od, placed):
        for _, px, py, pz, pw, ph, pd in placed:
            if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                return True
        return False

    def greedy_pack(sort_key, pos_generator, rng=None, perturbation=0.0):
        """Greedy packing with given sort key and position generator."""
        indexed_boxes = sorted(range(n), key=lambda i: sort_key(boxes[i]), reverse=True)

        placed = []  # List of (idx, x, y, z, w, h, d)
        total_vol = 0

        for idx in indexed_boxes:
            if time.time() - start_time > DEADLINE:
                break

            best_placement = None
            best_score = float('inf')

            for ow, oh, od in box_orientations[idx]:
                candidate_positions = pos_generator(placed, ow, oh, od)

                for cx, cy, cz in candidate_positions:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue

                    valid = True
                    for _, px, py, pz, pw, ph, pd in placed:
                        if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                            valid = False
                            break
                    if not valid:
                        continue

                    score = cx + cy + cz
                    if perturbation > 0 and rng is not None:
                        score += rng.random() * perturbation
                    if score < best_score:
                        best_score = score
                        best_placement = (idx, cx, cy, cz, ow, oh, od)

            if best_placement:
                placed.append(best_placement)
                total_vol += best_placement[4] * best_placement[5] * best_placement[6]

        result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        for idx, x, y, z, w, h, d in placed:
            result[idx] = (0, x, y, z, w, h, d)
        return result, total_vol

    def corners_generator(placed, ow, oh, od):
        corners = [(0, 0, 0)]
        for _, px, py, pz, pw, ph, pd in placed:
            corners.append((px + pw, py, pz))
            corners.append((px, py + ph, pz))
            corners.append((px, py, pz + pd))
            corners.append((px + pw, py + ph, pz))
            corners.append((px + pw, py, pz + pd))
            corners.append((px, py + ph, pz + pd))
        return corners

    def extended_corners_generator(placed, ow, oh, od):
        """More candidate positions including all 8 corners of each placed box."""
        corners = [(0, 0, 0)]
        for _, px, py, pz, pw, ph, pd in placed:
            corners.append((px + pw, py, pz))
            corners.append((px, py + ph, pz))
            corners.append((px, py, pz + pd))
            corners.append((px + pw, py + ph, pz))
            corners.append((px + pw, py, pz + pd))
            corners.append((px, py + ph, pz + pd))
            corners.append((px + pw, py + ph, pz + pd))
            corners.append((px + pw, py + ph, pz))
        return corners

    def shelf_generator(placed, ow, oh, od):
        """Shelf-based: try positions at top/side of each placed box plus corners."""
        positions = [(0, 0, 0)]
        for _, px, py, pz, pw, ph, pd in placed:
            positions.append((px, py, pz + pd))
            positions.append((px + pw, py, pz))
            positions.append((px, py + ph, pz))
            positions.append((px + pw, py + ph, pz))
            positions.append((px + pw, py, pz + pd))
            positions.append((px, py + ph, pz + pd))
            positions.append((px + pw, py + ph, pz + pd))
            positions.append((px + pw, py, pz))
            positions.append((px, py + ph, pz))
        seen = set()
        unique = []
        for p in positions:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def dense_positions_generator(placed, ow, oh, od):
        """Dense candidate positions from all unique coordinates."""
        xs = {0}
        ys = {0}
        zs = {0}
        for _, px, py, pz, pw, ph, pd in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        xs = sorted(set(x for x in xs if 0 <= x < W))
        ys = sorted(set(y for y in ys if 0 <= y < H))
        zs = sorted(set(z for z in zs if 0 <= z < D))
        
        if len(xs) * len(ys) * len(zs) > 5000:
            # Fall back to sparse
            return extended_corners_generator(placed, ow, oh, od)
        
        positions = [(x, y, z) for x in xs for y in ys for z in zs]
        return positions

    # Different sorting strategies
    sort_strategies = [
        lambda b: b[0]*b[1]*b[2],           # volume
        lambda b: max(b),                     # max dimension
        lambda b: sum(b),                     # sum of dimensions
        lambda b: max(b) / max(min(b), 1e-9),# elongation
        lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]),  # surface area
        lambda b: b[0]*b[1]*b[2] / max(b),   # volume per max dim
    ]

    pos_generators = [corners_generator, extended_corners_generator, shelf_generator, dense_positions_generator]

    best_result = None
    best_vol = 0

    # Deterministic strategies first
    for si, sort_key in enumerate(sort_strategies):
        if time.time() - start_time > DEADLINE:
            break
        for pi, pos_gen in enumerate(pos_generators):
            if time.time() - start_time > DEADLINE:
                break
            res, vol = greedy_pack(sort_key, pos_gen)
            if vol > best_vol:
                best_vol = vol
                best_result = res

    # Also try with small perturbation to explore nearby solutions
    for si, sort_key in enumerate(sort_strategies[:3]):
        if time.time() - start_time > DEADLINE:
            break
        for pi, pos_gen in enumerate(pos_generators[:2]):
            if time.time() - start_time > DEADLINE:
                break
            rng_pert = random.Random(si * 100 + pi)
            res, vol = greedy_pack(sort_key, pos_gen, rng=rng_pert, perturbation=0.5)
            if vol > best_vol:
                best_vol = vol
                best_result = res

    # Random restarts: shuffle box order with random tie-breaking
    if best_result is None:
        best_result = [(-1, 0, 0, 0, 0, 0, 0)] * n

    # Use random restarts with different shuffles
    rng = random.Random(42)
    restart_count = 0
    max_restarts = 80

    while time.time() - start_time < DEADLINE and restart_count < max_restarts:
        restart_count += 1

        # Create a random shuffle of box indices
        indices = list(range(n))
        rng.shuffle(indices)

        # Use a random sorting strategy with random perturbation
        sort_key = sort_strategies[rng.randint(0, len(sort_strategies) - 1)]
        pos_gen = pos_generators[rng.randint(0, len(pos_generators) - 1)]

        # Randomized greedy: when multiple positions are equally good, pick randomly
        indexed_boxes = sorted(indices, key=lambda i: sort_key(boxes[i]), reverse=True)

        placed = []
        total_vol = 0

        for idx in indexed_boxes:
            if time.time() - start_time > DEADLINE:
                break

            best_placement = None
            best_score = float('inf')

            for ow, oh, od in box_orientations[idx]:
                candidate_positions = pos_gen(placed, ow, oh, od)

                for cx, cy, cz in candidate_positions:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue

                    valid = True
                    for _, px, py, pz, pw, ph, pd in placed:
                        if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                            valid = False
                            break
                    if not valid:
                        continue

                    score = cx + cy + cz
                    # Add small random perturbation to break ties
                    score += rng.random() * 0.01

                    if score < best_score:
                        best_score = score
                        best_placement = (idx, cx, cy, cz, ow, oh, od)

            if best_placement:
                placed.append(best_placement)
                total_vol += best_placement[4] * best_placement[5] * best_placement[6]

        if total_vol > best_vol:
            best_vol = total_vol
            result = [(-1, 0, 0, 0, 0, 0, 0)] * n
            for idx, x, y, z, w, h, d in placed:
                result[idx] = (0, x, y, z, w, h, d)
            best_result = result

    # Final pass: try to insert unplaced boxes into the best solution
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        # Identify placed and unplaced boxes
        placed_indices = set()
        for i, (bin_id, x, y, z, w, h, d) in enumerate(best_result):
            if bin_id == 0:
                placed_indices.add(i)
        
        unplaced_indices = [i for i in range(n) if i not in placed_indices]
        
        if unplaced_indices:
            # Build placed list from best_result
            placed = []
            for i in placed_indices:
                _, x, y, z, w, h, d = best_result[i]
                placed.append((i, x, y, z, w, h, d))
            
            # Try to insert each unplaced box
            for idx in unplaced_indices:
                if time.time() - start_time > DEADLINE:
                    break
                
                best_placement = None
                best_score = float('inf')
                
                for ow, oh, od in box_orientations[idx]:
                    # Use dense positions for better chance of finding a fit
                    candidate_positions = dense_positions_generator(placed, ow, oh, od)
                    
                    for cx, cy, cz in candidate_positions:
                        if cx + ow > W or cy + oh > H or cz + od > D:
                            continue
                        
                        valid = True
                        for _, px, py, pz, pw, ph, pd in placed:
                            if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        score = cx + cy + cz
                        if score < best_score:
                            best_score = score
                            best_placement = (idx, cx, cy, cz, ow, oh, od)
                
                if best_placement:
                    placed.append(best_placement)
                    best_result[idx] = (0, best_placement[1], best_placement[2], best_placement[3], 
                                        best_placement[4], best_placement[5], best_placement[6])
                    best_vol += best_placement[4] * best_placement[5] * best_placement[6]

    return best_result
