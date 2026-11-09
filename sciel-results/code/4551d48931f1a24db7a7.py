import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 150s kill

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

    def greedy_pack(sort_key, seed=None):
        if seed is not None:
            random.seed(seed)

        indexed_boxes = sorted(enumerate(boxes), key=lambda x: sort_key(x[1]), reverse=True)

        placed = []  # List of (idx, x, y, z, w, h, d)
        corners = [(0, 0, 0)]
        best_vol = 0

        for idx, (w, h, d) in indexed_boxes:
            if time.time() - start_time > DEADLINE:
                break

            orientations = get_orientations(w, h, d)
            best_placement = None
            best_score = float('inf')

            for ow, oh, od in orientations:
                for cx, cy, cz in corners:
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
                best_vol += best_placement[4] * best_placement[5] * best_placement[6]

                _, bx, by, bz, bw, bh, bd = best_placement
                new_corners = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                ]
                for nc in new_corners:
                    if nc not in corners:
                        corners.append(nc)

        result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        for idx, x, y, z, w, h, d in placed:
            result[idx] = (0, x, y, z, w, h, d)

        return result, best_vol

    # Multiple sorting strategies
    strategies = [
        ("volume", lambda b: b[0] * b[1] * b[2]),
        ("max_dim", lambda b: max(b)),
        ("sum_dim", lambda b: sum(b)),
        ("min_dim", lambda b: min(b)),
        ("surface", lambda b: 2 * (b[0] * b[1] + b[1] * b[2] + b[0] * b[2])),
        ("vol_per_max", lambda b: b[0] * b[1] * b[2] / max(b) if max(b) > 0 else 0),
        ("ratio", lambda b: b[0] * b[1] / b[2] if b[2] > 0 else b[0] * b[1]),
    ]

    best_result = None
    best_vol = 0

    for name, key in strategies:
        if time.time() - start_time > DEADLINE:
            break
        res, vol = greedy_pack(key)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    return best_result
