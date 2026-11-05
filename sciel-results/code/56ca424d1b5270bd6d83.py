"""3D Container Loading - grid occupancy + best-fit + local search + multi-strategy."""
import time
import math

# EVOLVE-BLOCK-START

def _all_orientations(w, h, d):
    seen = set()
    result = []
    for o in ((w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)):
        if o not in seen:
            seen.add(o)
            result.append(o)
    return result


def _fits_grid(g, x, y, z, w, h, d):
    """Fast overlap check via 3D occupancy grid. True if free and in-bounds."""
    W, H, D = g
    if x < 0 or y < 0 or z < 0 or x + w > W or y + h > H or z + d > D:
        return False
    grid = g[3]
    for zz in range(z, z + d):
        row = grid[zz]
        for yy in range(y, y + h):
            r = row[yy]
            for xx in range(x, x + w):
                if r[xx]:
                    return False
    return True


def _mark(g, x, y, z, w, h, d):
    grid = g[3]
    for zz in range(z, z + d):
        row = grid[zz]
        for yy in range(y, y + h):
            r = row[yy]
            for xx in range(x, x + w):
                r[xx] = True


def _unmark(g, x, y, z, w, h, d):
    grid = g[3]
    for zz in range(z, z + d):
        row = grid[zz]
        for yy in range(y, y + h):
            r = row[yy]
            for xx in range(x, x + w):
                r[xx] = False


def _grid(W, H, D):
    return (W, H, D, [[[0] * W for _ in range(H)] for _ in range(D)])


def _place_best_fit(boxes, container, order, deadline, prefer_low=True):
    W, H, D = container
    g = _grid(W, H, D)
    placed = [None] * len(boxes)
    for idx in order:
        if time.time() > deadline:
            break
        ow, oh, od = boxes[idx]
        best = None
        for (w, h, d) in _all_orientations(ow, oh, od):
            if w > W or h > H or d > D:
                continue
            if prefer_low:
                # lowest free position (z, then y, then x)
                for z in range(D - d + 1):
                    for y in range(H - h + 1):
                        for x in range(W - w + 1):
                            if _fits_grid(g, x, y, z, w, h, d):
                                best = (idx, x, y, z, w, h, d)
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
            else:
                # best fit: maximize contact with already-placed boxes
                for z in range(D - d + 1):
                    for y in range(H - h + 1):
                        for x in range(W - w + 1):
                            if _fits_grid(g, x, y, z, w, h, d):
                                _mark(g, x, y, z, w, h, d)
                                contact = 0
                                # count occupied neighbors just outside the box
                                if x > 0:
                                    for yy in range(y, y + h):
                                        for zz in range(z, z + d):
                                            if g[3][zz][yy][x - 1]:
                                                contact += 1
                                if x + w < W:
                                    for yy in range(y, y + h):
                                        for zz in range(z, z + d):
                                            if g[3][zz][yy][x + w]:
                                                contact += 1
                                if y > 0:
                                    for xx in range(x, x + w):
                                        for zz in range(z, z + d):
                                            if g[3][zz][y - 1][xx]:
                                                contact += 1
                                if y + h < H:
                                    for xx in range(x, x + w):
                                        for zz in range(z, z + d):
                                            if g[3][zz][y + h][xx]:
                                                contact += 1
                                if z > 0:
                                    for xx in range(x, x + w):
                                        for yy in range(y, y + h):
                                            if g[3][z - 1][yy][xx]:
                                                contact += 1
                                if z + d < D:
                                    for xx in range(x, x + w):
                                        for yy in range(y, y + h):
                                            if g[3][z + d][yy][xx]:
                                                contact += 1
                                _unmark(g, x, y, z, w, h, d)
                                key = (-contact, z, y, x)
                                if best is None or key < best[0]:
                                    best = (key, idx, x, y, z, w, h, d)
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
        if best is not None:
            if len(best) == 8:
                _, idx, x, y, z, w, h, d = best
            else:
                idx, x, y, z, w, h, d = best
            placed[idx] = (idx, x, y, z, w, h, d)
            _mark(g, x, y, z, w, h, d)
    return [p for p in placed if p is not None]


def _local_search(placed, boxes, container, deadline):
    """Swap/insert repair: repeatedly try to improve volume by swapping placements."""
    W, H, D = container
    placed = list(placed)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(placed)
        for i in range(n):
            if time.time() > deadline:
                return placed
            idx_i, xi, yi, zi, wi, hi, di = placed[i]
            for j in range(i + 1, n):
                idx_j, xj, yj, zj, wj, hj, dj = placed[j]
                # try swapping the two boxes (with any orientation)
                g = _grid(W, H, D)
                for (p, x, y, z, w, h, d) in placed:
                    if p != i and p != j:
                        _mark(g, x, y, z, w, h, d)
                # place box i in box j's old spot region and box j in box i's region
                ok_i = None
                ok_j = None
                for (w1, h1, d1) in _all_orientations(boxes[idx_i]):
                    for (w2, h2, d2) in _all_orientations(boxes[idx_j]):
                        # box j at its old position
                        if not _fits_grid(g, xj, yj, zj, w2, h2, d2):
                            continue
                        _mark(g, xj, yj, zj, w2, h2, d2)
                        # find spot for box i (prefer low)
                        for z in range(D - d1 + 1):
                            for y in range(H - h1 + 1):
                                for x in range(W - w1 + 1):
                                    if _fits_grid(g, x, y, z, w1, h1, d1):
                                        ok_i = (x, y, z, w1, h1, d1)
                                        break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        _unmark(g, xj, yj, zj, w2, h2, d2)
                        if ok_i is not None:
                            ok_j = (xj, yj, zj, w2, h2, d2)
                            break
                    if ok_i is not None:
                        break
                if ok_i is not None:
                    placed[i] = (idx_i, ok_i[0], ok_i[1], ok_i[2], ok_i[3], ok_i[4], ok_i[5])
                    placed[j] = (idx_j, ok_j[0], ok_j[1], ok_j[2], ok_j[3], ok_j[4], ok_j[5])
                    improved = True
                    break
            if improved:
                break
    return placed


def _vol(placed):
    return sum(w * h * d for (_, _, _, _, w, h, d) in placed)


def _verify(placed, boxes, container):
    W, H, D = container
    n = len(boxes)
    seen = set()
    for (idx, x, y, z, w, h, d) in placed:
        if idx < 0 or idx >= n or idx in seen:
            return False
        seen.add(idx)
        if x < 0 or y < 0 or z < 0 or x + w > W or y + h > H or z + d > D:
            return False
    for i in range(len(placed)):
        _, x1, y1, z1, w1, h1, d1 = placed[i]
        for j in range(i + 1, len(placed)):
            _, x2, y2, z2, w2, h2, d2 = placed[j]
            if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1):
                return False
    return True


def _orders(boxes):
    n = len(boxes)
    idxs = list(range(n))
    def key_vol(b): return -b[0] * b[1] * b[2]
    def key_max(b): return -max(b)
    def key_min(b): return -min(b)
    def key_sum(b): return -(b[0] + b[1] + b[2])
    def key_surf(b): return -(2 * (b[0] * b[1] + b[1] * b[2] + b[0] * b[2]))
    def key_ratio(b): return -(b[0] * b[1] * b[2]) / (min(b) + 1)
    def key_flat(b): return -(max(b) / (min(b) + 1))
    orders = []
    for k in (key_vol, key_max, key_min, key_sum, key_surf, key_ratio, key_flat):
        o = sorted(idxs, key=lambda i: k(boxes[i]))
        orders.append(o)
    return orders


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start = time.time()
    hard_deadline = start + 138.0

    best = None
    best_vol = 0

    # Build candidate orders
    orders = _orders(boxes)

    # Run strategies: for each order, try prefer_low and best_fit
    def run(order, prefer_low, dl):
        return _place_best_fit(boxes, container, order, dl, prefer_low)

    # Time budget allocation
    # primary: first order, prefer_low, generous
    primary_dl = start + 55.0
    for oi, order in enumerate(orders):
        if time.time() > hard_deadline - 8:
            break
        # primary gets more time
        if oi == 0:
            dl = min(primary_dl, hard_deadline - 30)
            placed = run(order, True, dl)
            if _verify(placed, boxes, container):
                v = _vol(placed)
                if v > best_vol:
                    best_vol, best = v, placed
            # also best_fit on same order with short budget
            if time.time() < hard_deadline - 20:
                placed2 = run(order, False, min(time.time() + 12, hard_deadline - 18))
                if _verify(placed2, boxes, container):
                    v = _vol(placed2)
                    if v > best_vol:
                        best_vol, best = v, placed2
        else:
            dl = min(time.time() + 14, hard_deadline - 12)
            placed = run(order, True, dl)
            if _verify(placed, boxes, container):
                v = _vol(placed)
                if v > best_vol:
                    best_vol, best = v, placed

    # Local search on the best result with remaining time
    if best is not None and time.time() < hard_deadline - 5:
        ls_dl = hard_deadline - 3
        repaired = _local_search(best, boxes, container, ls_dl)
        if _verify(repaired, boxes, container):
            v = _vol(repaired)
            if v > best_vol:
                best_vol, best = v, repaired

    if best is None:
        return [(-1, 0, 0, 0, 0, 0, 0)] * n

    output = [(-1, 0, 0, 0, 0, 0, 0)] * n
    for (idx, x, y, z, w, h, d) in best:
        if 0 <= idx < n:
            output[idx] = (0, x, y, z, w, h, d)
    return output

# EVOLVE-BLOCK-END
