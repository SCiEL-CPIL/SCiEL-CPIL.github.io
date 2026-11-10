# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k_const = 3

    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul_f(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k_const) % p,
                (a[0] * b[1] + a[1] * b[0]) % p)

    field_elems = [(a, b) for a in range(p) for b in range(p)]
    all_points = [(x, y) for x in field_elems for y in field_elems]
    n_pts = len(all_points)
    pt_idx = {pt: i for i, pt in enumerate(all_points)}

    directions = [((1, 0), m) for m in field_elems] + [((0, 1), (0, 0))]
    n_dirs = len(directions)

    def sq(x):
        return mul_f(x, x)

    parabola = set()
    for x in field_elems:
        parabola.add((x, sq(x)))

    in_S = np.zeros(n_pts, dtype=bool)
    for pt in parabola:
        in_S[pt_idx[pt]] = True

    # count[x, v] = # points on line(x,v) \ {x} that are NOT in S
    count = np.full((n_pts, n_dirs), q - 1, dtype=np.int32)

    # Precompute: for each point p, for each direction v, the list of other points
    # on the line through p with direction v.
    # When p is added to S, for each such x, count[x, v] decreases by 1.
    line_others = [[[] for _ in range(n_dirs)] for _ in range(n_pts)]
    for pi in range(n_pts):
        pt = all_points[pi]
        for vi in range(n_dirs):
            vx, vy = directions[vi]
            for t in field_elems:
                if t == (0, 0):
                    continue
                tx = mul_f(t, vx)
                ty = mul_f(t, vy)
                qx = add_f(pt[0], tx)
                qy = add_f(pt[1], ty)
                qpt = (qx, qy)
                line_others[pi][vi].append(pt_idx[qpt])

    # Initialize counts by subtracting for each parabola point
    for pt in parabola:
        pi = pt_idx[pt]
        for vi in range(n_dirs):
            for xi in line_others[pi][vi]:
                count[xi, vi] -= 1

    valid_cnt = (count > 0).sum(axis=1)
    assert (valid_cnt > 0).all()

    # affected[pi] = list of (xi, vi) that change when pi is added to S
    affected = [[] for _ in range(n_pts)]
    for pi in range(n_pts):
        for vi in range(n_dirs):
            for xi in line_others[pi][vi]:
                affected[pi].append((xi, vi))

    deadline = time.time() + 570
    remaining = [i for i in range(n_pts) if not in_S[i]]

    # Greedy: repeatedly try to add points. Use a priority based on how "safe" the addition is.
    # For simplicity, do passes over remaining points in order, adding if feasible.
    
    for _pass in range(50):
        if time.time() > deadline:
            break
        progress = False
        for pi in remaining[:]:
            if in_S[pi]:
                continue
            # Check feasibility
            # Group affected changes by xi
            from collections import defaultdict
            ch_by_x = defaultdict(list)
            for (xi, vi) in affected[pi]:
                ch_by_x[xi].append(vi)
            
            ok = True
            for xi, vis in ch_by_x.items():
                lost = 0
                for vi in vis:
                    if count[xi, vi] > 0 and count[xi, vi] - 1 == 0:
                        lost += 1
                if valid_cnt[xi] - lost <= 0:
                    ok = False
                    break
            
            if ok:
                # Apply
                for (xi, vi) in affected[pi]:
                    if count[xi, vi] > 0 and count[xi, vi] - 1 == 0:
                        valid_cnt[xi] -= 1
                    count[xi, vi] -= 1
                in_S[pi] = True
                remaining.remove(pi)
                progress = True
        if not progress:
            break

    pts = []
    for i in range(n_pts):
        if not in_S[i]:
            pt = all_points[i]
            pts.append([[pt[0][0], pt[0][1]], [pt[1][0], pt[1][1]]])
    return np.array(pts, dtype=np.int64)

# EVOLVE-BLOCK-END
