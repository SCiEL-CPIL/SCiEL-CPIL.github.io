"""Rewritten 3D bin packer: fast best-fit-decreasing with efficient corner
search, multi-strategy sorting, and time-bounded consolidation to minimize
bin count B and maximize last-bin fill."""
import time
import random

_DEADLINE = 140.0  # safety margin before the 150s kill


def _overlap(a, b):
    # a, b: (x, y, z, w, h, d)
    ax, ay, az, aw, ah, ad = a
    bx, by, bz, bw, bh, bd = b
    if ax + aw <= bx or bx + bw <= ax:
        return False
    if ay + ah <= by or by + bh <= ay:
        return False
    if az + ad <= bz or bz + bd <= az:
        return False
    return True


def _find_corner(box, items, W, H, D):
    """Find the lowest (z, then y, then x) corner where box fits without overlap."""
    w, h, d = box
    # Candidate coordinates: 0 and right/top/back edges of existing items
    xs = sorted(set([0] + [it[0] + it[3] for it in items if it[0] + it[3] + w <= W]))
    ys = sorted(set([0] + [it[1] + it[4] for it in items if it[1] + it[4] + h <= H]))
    zs = sorted(set([0] + [it[2] + it[5] for it in items if it[2] + it[5] + d <= D]))

    for z in zs:
        for y in ys:
            for x in xs:
                # Check overlap with all items
                ok = True
                for it in items:
                    if _overlap((x, y, z, w, h, d), it):
                        ok = False
                        break
                if ok:
                    return (x, y, z)
    return None


def _best_bin(box, bins, W, H, D):
    """Return (bin_index, x, y, z) of the best fit for box, or None.
    Best-fit: prefer bin with highest current volume (fill it up),
    tie-break by lower z, then y, then x."""
    w, h, d = box
    if w > W or h > H or d > D:
        return None

    best = None
    best_key = None
    for bi, items in enumerate(bins):
        placed = _find_corner(box, items, W, H, D)
        if placed is None:
            continue
        x, y, z = placed
        vol = sum(a[3] * a[4] * a[5] for a in items)
        key = (-vol, z, y, x)
        if best_key is None or key < best_key:
            best_key = key
            best = (bi, x, y, z)
    return best


def _compute_score(placements, container):
    W, H, D = container
    cvol = W * H * D
    if not placements:
        return 0.0
    bins = {}
    last = -1
    for bi, x, y, z, w, h, d in placements:
        bins.setdefault(bi, 0)
        bins[bi] += w * h * d
        if bi > last:
            last = bi
    B = len(bins)
    last_vol = bins.get(last, 0)
    return B + 0.1 * (1 - last_vol / cvol)


def _pack_with_order(boxes, container, order):
    """Greedy best-fit placement with a given box order."""
    W, H, D = container
    n = len(boxes)
    bins = []
    pos = [None] * n

    for idx in order:
        box = boxes[idx]
        w, h, d = box
        res = _best_bin(box, bins, W, H, D)
        if res is not None:
            bi, x, y, z = res
            bins[bi].append((x, y, z, w, h, d))
            pos[idx] = (bi, x, y, z, w, h, d)
        else:
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            pos[idx] = (bi, 0, 0, 0, w, h, d)
    return pos, bins


def _consolidate(pos, bins, boxes, container, t0, deadline):
    """Try to move boxes from the last bin into earlier bins to reduce B."""
    W, H, D = container
    n = len(boxes)

    def current_score():
        return _compute_score([pos[i] for i in range(n)], container)

    best_pos = [p[:] if isinstance(p, tuple) else list(p) for p in pos]
    best_s = current_score()

    def try_move(box_idx, target_bin):
        bi, x, y, z, w, h, d = pos[box_idx]
        if target_bin == bi:
            return False
        cur = bins[bi]
        # Find and remove the item
        item_to_remove = None
        for it in cur:
            if it[0] == x and it[1] == y and it[2] == z and it[3] == w and it[4] == h and it[5] == d:
                item_to_remove = it
                break
        if item_to_remove is None:
            return False
        cur.remove(item_to_remove)
        # Try to place in target bin
        placed = _find_corner(boxes[box_idx], bins[target_bin], W, H, D)
        if placed is None:
            cur.append(item_to_remove)
            return False
        nx, ny, nz = placed
        bins[target_bin].append((nx, ny, nz, w, h, d))
        pos[box_idx] = (target_bin, nx, ny, nz, w, h, d)
        return True

    attempts = 0
    while time.time() - t0 < deadline - 5.0 and attempts < 5000:
        attempts += 1
        used_bins = sorted(set(b for b, *_ in pos))
        if len(used_bins) <= 1:
            break
        last_bin = used_bins[-1]
        cand = [i for i in range(n) if pos[i][0] == last_bin]
        if not cand:
            break
        random.shuffle(cand)
        moved = False
        for i in cand:
            for tb in used_bins[:-1]:
                if try_move(i, tb):
                    s = current_score()
                    if s < best_s:
                        best_s = s
                        best_pos = [list(p) for p in pos]
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break

    return best_pos, best_s


def pack(boxes, container):
    W, H, D = container
    t0 = time.time()
    n = len(boxes)

    # Try multiple sorting strategies and keep the best
    strategies = []

    # 1. Volume descending
    strategies.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    # 2. Max dimension descending
    strategies.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    # 3. Surface area descending
    strategies.append(sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))

    best_pos = None
    best_score = float('inf')

    for order in strategies:
        if time.time() - t0 > _DEADLINE - 10:
            break
        pos, bins = _pack_with_order(boxes, container, order)
        s = _compute_score([pos[i] for i in range(n)], container)
        if s < best_score:
            best_score = s
            best_pos = [list(p) for p in pos]
            # Save bins for consolidation
            best_bins = bins

    # Run consolidation on the best solution
    if best_pos is not None and time.time() - t0 < _DEADLINE - 5:
        # Rebuild bins from best_pos for consolidation
        bins_rebuilt = {}
        for i in range(n):
            bi, x, y, z, w, h, d = best_pos[i]
            if bi not in bins_rebuilt:
                bins_rebuilt[bi] = []
            bins_rebuilt[bi].append((x, y, z, w, h, d))
        # Convert to list indexed by bin number
        max_bin = max(bins_rebuilt.keys()) if bins_rebuilt else 0
        bins_list = [bins_rebuilt.get(b, []) for b in range(max_bin + 1)]
        pos_list = [tuple(best_pos[i]) for i in range(n)]
        improved_pos, improved_s = _consolidate(pos_list, bins_list, boxes, container, t0, _DEADLINE)
        if improved_s < best_score:
            best_score = improved_s
            best_pos = improved_pos

    # Return placements in original order
    return [tuple(best_pos[i]) for i in range(n)]
