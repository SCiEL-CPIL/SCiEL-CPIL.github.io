# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 620.0
START_TIME = time.time()


def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)


def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3
    n_total = 2401

    # Point encoding: idx = ((xa * 7 + xb) * 7 + ya) * 7 + yb
    def int_to_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return ((xa, xb), (ya, yb))

    def point_to_int(pt):
        (xa, xb), (ya, yb) = pt
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    def field_mul(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        a3 = (a1 * a2 + k * b1 * b2) % p
        b3 = (a1 * b2 + a2 * b1) % p
        return (a3, b3)

    def field_sub(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 - a2) % p, (b1 - b2) % p)

    # Directions: (1, m) for all m in F_49, plus (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 1), (0, 0)))
    n_dirs = len(directions)  # 50

    # Precompute for each direction and each point, the other points on the line
    # line_others[d][i] = list of other point indices on same line through i in direction d
    line_others = []
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]
        line_map = {}
        for pt_idx in range(n_total):
            pt = int_to_point(pt_idx)
            (x, y) = pt
            if vx == (1, 0):
                mx = field_mul(vy, x)
                c = field_sub(y, mx)
            else:
                c = x
            if c not in line_map:
                line_map[c] = []
            line_map[c].append(pt_idx)
        
        neigh = [[] for _ in range(n_total)]
        for c, pts_on_line in line_map.items():
            for i in pts_on_line:
                neigh[i] = [j for j in pts_on_line if j != i]
        line_others.append(neigh)

    # Base construction: unital-based
    squares = set((u * u) % p for u in range(p))
    S = set()
    for pt_idx in range(n_total):
        pt = int_to_point(pt_idx)
        (xa, xb), (ya, yb) = pt
        in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            S.add(pt_idx)

    # Track valid directions for each point
    # valid_dirs[x] = set of direction indices that are valid for point x
    valid_dirs = [set() for _ in range(n_total)]
    for d_idx in range(n_dirs):
        for pt_idx in range(n_total):
            ok = True
            for other in line_others[d_idx][pt_idx]:
                if other in S:
                    ok = False
                    break
            if ok:
                valid_dirs[pt_idx].add(d_idx)

    # Greedily add points to S
    candidates = [i for i in range(n_total) if i not in S]
    np.random.seed(42)
    np.random.shuffle(candidates)

    added_count = 0
    for cand in candidates:
        if _time_left() < 150:
            break

        # Check if cand has a valid direction (considering current S)
        cand_valid = False
        for d_idx in range(n_dirs):
            ok = True
            for other in line_others[d_idx][cand]:
                if other in S:
                    ok = False
                    break
            if ok:
                cand_valid = True
                break

        if not cand_valid:
            continue

        # Check if adding cand would invalidate any other point
        # For each direction d, cand is on lines with other points
        # For each other point x on line(cand, d), if d is in valid_dirs[x] and 
        # d is the ONLY valid direction for x, then x would lose all valid directions
        valid_add = True
        for d_idx in range(n_dirs):
            for other in line_others[d_idx][cand]:
                if other in S:
                    continue
                if d_idx in valid_dirs[other]:
                    if len(valid_dirs[other]) == 1:
                        valid_add = False
                        break
            if not valid_add:
                break

        if not valid_add:
            continue

        # Add cand to S
        S.add(cand)
        added_count += 1

        # Update valid_dirs for affected points
        for d_idx in range(n_dirs):
            for other in line_others[d_idx][cand]:
                if other in S:
                    continue
                if d_idx in valid_dirs[other]:
                    valid_dirs[other].discard(d_idx)

    # Build final result
    points = []
    for pt_idx in range(n_total):
        if pt_idx not in S:
            pt = int_to_point(pt_idx)
            (xa, xb), (ya, yb) = pt
            points.append([[xa, xb], [ya, yb]])

    arr = np.array(points, dtype=np.int64)
    return arr


# EVOLVE-BLOCK-END
