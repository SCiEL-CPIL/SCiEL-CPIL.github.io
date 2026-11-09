"""3D bin packer via simulated annealing over box permutations.

Structural change: instead of committing to a single greedy order and then
patching the result with local moves, we treat the *order* in which boxes are
placed as the optimization variable. Each candidate solution is produced by a
fast deterministic best-fit placement of the whole sequence. The search is a
simulated annealing walk over adjacent-swap / 2-opt moves on that sequence,
accepting worse solutions with a temperature schedule so it can escape the
local minima that trap a pure greedy heuristic. The objective is the exact
task score, so the search directly optimizes B and last-bin fill.
"""
import time
import random

_DEADLINE = 135.0  # safety margin before the 150s kill


def _fits(x, y, z, w, h, d, items, W, H, D):
    if x + w > W or y + h > H or z + d > D:
        return False
    for (ix, iy, iz, iw, ih, id_) in items:
        if not (x + w <= ix or ix + iw <= x or
                y + h <= iy or iy + ih <= y or
                z + d <= iz or iz + id_ <= z):
            return False
    return True


def _place(box, items, W, H, D):
    """Return (x, y, z) of the lowest-then-leftmost corner where box fits, else None.
    Candidate coords: 0 plus the far faces of existing items."""
    w, h, d = box
    xs = sorted(set([0] + [it[0] + it[3] for it in items if it[0] + it[3] + w <= W]))
    ys = sorted(set([0] + [it[1] + it[4] for it in items if it[1] + it[4] + h <= H]))
    zs = sorted(set([0] + [it[2] + it[5] for it in items if it[2] + it[5] + d <= D]))
    for z in zs:
        for y in ys:
            for x in xs:
                if _fits(x, y, z, w, h, d, items, W, H, D):
                    return (x, y, z)
    return None


def _score_of(pos, n, cvol):
    """Compute task score from a placement array pos[i] = (bin, x, y, z, w, h, d)."""
    bin_vol = {}
    last = -1
    for p in pos:
        bi = p[0]
        bin_vol[bi] = bin_vol.get(bi, 0) + p[3] * p[4] * p[5]
        if bi > last:
            last = bi
    B = len(bin_vol)
    last_vol = bin_vol.get(last, 0)
    return B + 0.1 * (1 - last_vol / cvol)


def _pack_order(boxes, container, order):
    """Deterministic best-fit placement of the given box order.
    Returns (pos, bins, score)."""
    W, H, D = container
    cvol = W * H * D
    n = len(boxes)
    bins = []
    pos = [None] * n
    for idx in order:
        w, h, d = boxes[idx]
        placed = None
        best_key = None
        for bi, items in enumerate(bins):
            p = _place((w, h, d), items, W, H, D)
            if p is None:
                continue
            x, y, z = p
            vol = sum(a[3] * a[4] * a[5] for a in items)
            key = (-vol, z, y, x)
            if best_key is None or key < best_key:
                best_key = key
                placed = (bi, x, y, z)
        if placed is not None:
            bi, x, y, z = placed
            bins[bi].append((x, y, z, w, h, d))
            pos[idx] = (bi, x, y, z, w, h, d)
        else:
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            pos[idx] = (bi, 0, 0, 0, w, h, d)
    s = _score_of(pos, n, cvol)
    return pos, bins, s


def pack(boxes, container):
    W, H, D = container
    cvol = W * H * D
    t0 = time.time()
    n = len(boxes)

    # ---- Build a diverse set of initial orders and evaluate them ----
    base_vol = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    base_max = [max(boxes[i]) for i in range(n)]

    orders = []
    orders.append(sorted(range(n), key=lambda i: base_vol[i], reverse=True))
    orders.append(sorted(range(n), key=lambda i: base_max[i], reverse=True))
    orders.append(sorted(range(n), key=lambda i: base_vol[i] + 0.5 * base_max[i], reverse=True))
    # a few randomized perturbations of the volume order for diversity
    for _ in range(3):
        o = list(orders[0])
        random.shuffle(o)
        # bias: keep large boxes toward the front
        o.sort(key=lambda i: base_vol[i], reverse=True)
        # swap a few random pairs
        for _ in range(n // 4):
            a = random.randrange(n)
            b = random.randrange(n)
            o[a], o[b] = o[b], o[a]
        orders.append(o)

    best_pos = None
    best_score = float('inf')
    best_order = None
    for order in orders:
        if time.time() - t0 > _DEADLINE - 8:
            break
        pos, bins, s = _pack_order(boxes, container, order)
        if s < best_score:
            best_score = s
            best_pos = pos
            best_order = order[:]

    # ---- Simulated annealing over the order (adjacent swap / 2-opt moves) ----
    if best_order is not None and time.time() - t0 < _DEADLINE - 5:
        cur_order = best_order[:]
        cur_score = best_score
        T0 = 0.05
        T_end = 0.001
        # Determine number of steps from remaining time budget
        remaining = _DEADLINE - 5 - (time.time() - t0)
        # Estimate cost per full re-pack is ~ (n * bins * corners * items).
        # We'll just run until the deadline, checking each iteration.
        steps = 0
        max_steps = 200000
        while time.time() - t0 < _DEADLINE - 3 and steps < max_steps:
            steps += 1
            if n < 2:
                break
            # Pick a random move: adjacent swap or 2-opt segment reversal
            if random.random() < 0.5:
                a = random.randrange(n)
                b = a + random.randrange(1, min(n - a, 6))
                new_order = cur_order[:]
                new_order[a], new_order[b] = new_order[b], new_order[a]
            else:
                a = random.randrange(n)
                b = random.randrange(a + 1, n)
                new_order = cur_order[:]
                new_order[a:b + 1] = reversed(new_order[a:b + 1])

            pos, bins, s = _pack_order(boxes, container, new_order)
            delta = s - cur_score
            # temperature schedule (geometric cooling)
            frac = steps / max_steps
            T = T0 * (T_end / T0) ** frac
            if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-9)):
                cur_order = new_order
                cur_score = s
                cur_pos = pos
                if s < best_score:
                    best_score = s
                    best_pos = pos
                    best_order = cur_order[:]

    # Final: return placements in original index order
    return [tuple(best_pos[i]) for i in range(n)]
