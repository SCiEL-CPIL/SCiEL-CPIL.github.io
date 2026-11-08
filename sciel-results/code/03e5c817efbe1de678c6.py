"""Rewritten 3D bin packer: best-fit into existing bins, with a light
time-bounded local-search improvement. Minimizes number of bins (B),
then maximizes last-bin fill."""
import time
import random
import numpy as np

_DEADLINE = 110.0  # safety margin before the 120s kill


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


def _fits(box, items, W, H, D):
    w, h, d = box
    for it in items:
        if _overlap((0, 0, 0, w, h, d), it):
            return False
    return True


def _best_bin(box, bins, W, H, D):
    """Return (bin_index, x, y, z) of the best fit for box, or None."""
    w, h, d = box
    best = None
    best_vol = None
    for bi, items in enumerate(bins):
        # quick rejection: if box alone doesn't fit container dims
        if w > W or h > H or d > D:
            continue
        # find the lowest corner that fits
        placed = _find_corner(box, items, W, H, D)
        if placed is None:
            continue
        x, y, z = placed
        # prefer the bin with the highest existing volume (fill it up),
        # tie-break by lower z, then y, then x
        vol = sum(a[3] * a[4] * a[5] for a in items)
        key = (-vol, z, y, x)
        if best_vol is None or key < best_vol:
            best_vol = key
            best = (bi, x, y, z)
    return best


def _find_corner(box, items, W, H, D):
    """Find lowest (z, then y, then x) corner where box fits without overlap."""
    w, h, d = box
    # candidate x: 0 and right edges of existing items
    xs = sorted(set([0] + [it[0] + it[3] for it in items if it[0] + it[3] <= W]))
    ys = sorted(set([0] + [it[1] + it[4] for it in items if it[1] + it[4] <= H]))
    zs = sorted(set([0] + [it[2] + it[5] for it in items if it[2] + it[5] <= D]))
    for z in zs:
        if z + d > D:
            break
        for y in ys:
            if y + h > H:
                break
            for x in xs:
                if x + w > W:
                    break
                if _fits(box, items, W, H, D):
                    pass
                # check actual fit at (x,y,z)
                ok = True
                for it in items:
                    if _overlap((x, y, z, w, h, d), it):
                        ok = False
                        break
                if ok:
                    return (x, y, z)
    return None


def _compute_score(placements, boxes, container):
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


def pack(boxes, container):
    W, H, D = container
    t0 = time.time()
    n = len(boxes)

    # Sort by volume descending (larger first) for better packing,
    # but remember original indices to restore order.
    order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],
                   reverse=True)

    bins = []  # list of lists of (x,y,z,w,h,d)
    # placement per original index
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
            # open a new bin
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            pos[idx] = (bi, 0, 0, 0, w, h, d)

    # Build a helper to recompute placements from pos
    def make_placements():
        return [pos[i] for i in range(n)]

    # Local search: try moving a box to a different bin/corner to reduce B
    # or improve last-bin fill. Keep it very light and time-bounded.
    def current_score():
        return _compute_score(make_placements(), boxes, container)

    best = make_placements()
    best_s = current_score()

    # Simple improvement: for each box (small first), try to move it into
    # an earlier bin to possibly free the last bin.
    def try_move(box_idx, target_bin):
        """Try to relocate box_idx into target_bin (a different bin)."""
        bi, x, y, z, w, h, d = pos[box_idx]
        if target_bin == bi:
            return False
        # remove from current bin
        cur = bins[bi]
        item = None
        for k, it in enumerate(cur):
            if it[3] == w and it[4] == h and it[5] == d and it[0] == x and it[1] == y and it[2] == z:
                item = it
                break
        if item is None:
            return False
        # temporarily remove
        cur.remove(item)
        # find a corner in target bin
        placed = _find_corner(boxes[box_idx], bins[target_bin], W, H, D)
        if placed is None:
            cur.append(item)
            return False
        nx, ny, nz = placed
        bins[target_bin].append((nx, ny, nz, w, h, d))
        pos[box_idx] = (target_bin, nx, ny, nz, w, h, d)
        return True

    # Attempt a few relocations to reduce bin count
    attempts = 0
    while time.time() - t0 < _DEADLINE - 5.0 and attempts < 2000:
        attempts += 1
        # current bin count
        used_bins = sorted(set(b for b, *_ in pos))
        if len(used_bins) <= 1:
            break
        # try to move a box from the highest-numbered bin into an earlier one
        last_bin = used_bins[-1]
        # find boxes in last bin
        cand = [i for i in range(n) if pos[i][0] == last_bin]
        random.shuffle(cand)
        moved = False
        for i in cand:
            for tb in used_bins[:-1]:
                if try_move(i, tb):
                    # check if last bin is now empty -> B reduced
                    still = any(pos[j][0] == last_bin for j in range(n))
                    s = current_score()
                    if s < best_s:
                        best_s = s
                        best = make_placements()
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break

    # Ensure all boxes are placed (they are, since we only moved existing ones)
    return best
