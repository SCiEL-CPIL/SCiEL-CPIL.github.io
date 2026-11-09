import time
import itertools

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True

    def get_orientations(w, h, d):
        perms = set(itertools.permutations([w, h, d]))
        return list(perms)

    def greedy_pack(sort_key, use_extended_corners=False):
        indexed_boxes = sorted(enumerate(boxes), key=lambda x: sort_key(x[1]), reverse=True)

        placed = []  # List of (idx, x, y, z, w, h, d)
        corners = [(0, 0, 0)]
        best_vol = 0

        for idx, (bw, bh, bd) in indexed_boxes:
            orientations = get_orientations(bw, bh, bd)
            best_placement = None
            best_score = float('inf')

            for ow, oh, od in orientations:
                # Generate candidate positions
                candidate_positions = [(0, 0, 0)]
                for _, px, py, pz, pw, ph, pd in placed:
                    candidate_positions.append((px + pw, py, pz))
                    candidate_positions.append((px, py + ph, pz))
                    candidate_positions.append((px, py, pz + pd))
                    if use_extended_corners:
                        candidate_positions.append((px + pw, py + ph, pz))
                        candidate_positions.append((px + pw, py, pz + pd))
                        candidate_positions.append((px, py + ph, pz + pd))
                        candidate_positions.append((px + pw, py + ph, pz + pd))

                # Deduplicate while preserving order
                seen = set()
                unique_positions = []
                for pos in candidate_positions:
                    if pos not in seen:
                        seen.add(pos)
                        unique_positions.append(pos)

                for cx, cy, cz in unique_positions:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue

                    valid = True
                    for _, px, py, pz, pw, ph, pd in placed:
                        if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                            valid = False
                            break
                    if not valid:
                        continue

                    # Score: primary is distance to origin, secondary is how "tight" the fit is
                    # Prefer positions that are closer to origin and also prefer orientations
                    # that align with container axes more tightly
                    score = cx + cy + cz
                    # Small penalty for larger gaps in any dimension to encourage tighter packing
                    # This helps fill corners better
                    gap_penalty = 0.0
                    if cx > 0:
                        gap_penalty += 0.1 * cx
                    if cy > 0:
                        gap_penalty += 0.1 * cy
                    if cz > 0:
                        gap_penalty += 0.1 * cz
                    score += gap_penalty

                    if score < best_score:
                        best_score = score
                        best_placement = (idx, cx, cy, cz, ow, oh, od)

            if best_placement:
                placed.append(best_placement)
                best_vol += best_placement[4] * best_placement[5] * best_placement[6]

        result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        for idx, x, y, z, w, h, d in placed:
            result[idx] = (0, x, y, z, w, h, d)

        return result, best_vol

    # Strategy 1: Sort by Volume, basic corners
    res1, vol1 = greedy_pack(lambda b: b[0] * b[1] * b[2], False)
    if time.time() - start_time > DEADLINE:
        return res1

    # Strategy 2: Sort by Max Dimension, extended corners
    res2, vol2 = greedy_pack(lambda b: max(b), True)
    if time.time() - start_time > DEADLINE:
        return res2 if vol2 > vol1 else res1

    # Strategy 3: Sort by Surface Area, extended corners
    res3, vol3 = greedy_pack(lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]), True)
    if time.time() - start_time > DEADLINE:
        best_res = res1
        best_vol = vol1
        if vol2 > best_vol:
            best_res = res2
            best_vol = vol2
        if vol3 > best_vol:
            best_res = res3
            best_vol = vol3
        return best_res

    # Strategy 4: Sort by Elongation, extended corners
    res4, vol4 = greedy_pack(lambda b: max(b) / max(min(b), 1e-9), True)
    if time.time() - start_time > DEADLINE:
        best_res = res1
        best_vol = vol1
        if vol2 > best_vol:
            best_res = res2
            best_vol = vol2
        if vol3 > best_vol:
            best_res = res3
            best_vol = vol3
        if vol4 > best_vol:
            best_res = res4
            best_vol = vol4
        return best_res

    # Strategy 5: Sort by min dimension (square-ish first), extended corners
    res5, vol5 = greedy_pack(lambda b: min(b), True)

    best_res = res1
    best_vol = vol1
    for res, vol in [(res2, vol2), (res3, vol3), (res4, vol4), (res5, vol5)]:
        if vol > best_vol:
            best_res = res
            best_vol = vol

    return best_res
