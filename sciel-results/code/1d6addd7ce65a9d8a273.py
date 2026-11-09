"""3D bin packing: maximal-rectangles (skyline) region splitting with multi-ordering search."""
import time
import itertools
import math

# EVOLVE-BLOCK-START
def _split_regions(placed):
    """Build list of maximal free boxes from a list of placed (x,y,z,w,h,d)."""
    regions = []
    for x, y, z, w, h, d in placed:
        for (rx, ry, rz, rw, rh, rd) in list(regions):
            new = []
            # Does placed intersect this region?
            if x < rx + rw and x + w > rx and y < ry + rh and y + h > ry and z < rz + rd and z + d > rz:
                # Split region around the placed box
                if rx < x:
                    new.append((rx, ry, rz, x - rx, rh, rd))
                if x + w < rx + rw:
                    new.append((x + w, ry, rz, rx + rw - (x + w), rh, rd))
                if ry < y:
                    new.append((rx, ry, rz, rw, y - ry, rd))
                if y + h < ry + rh:
                    new.append((rx, y + h, rz, rw, ry + rh - (y + h), rd))
                if rz < z:
                    new.append((rx, ry, rz, rw, rh, z - rz))
                if z + d < rz + rd:
                    new.append((rx, ry, z + d, rw, rh, rz + rd - (z + d)))
            else:
                new.append((rx, ry, rz, rw, rh, rd))
            regions.remove((rx, ry, rz, rw, rh, rd))
            regions.extend(new)
    return regions


def _try_place_order(boxes, order, W, H, D, deadline):
    """Place boxes in given order using maximal-rectangles; return placements or None."""
    t0 = time.time()
    bins = []  # each bin: list of (x,y,z,w,h,d)
    placements = []
    box_idx = {id: i for i, id in enumerate(order)}
    for bi, (w, h, d) in enumerate(boxes):
        if time.time() > deadline:
            return None
        best = None  # (score, bin_idx, x, y, z)
        # Try existing bins
        for bi_idx, placed in enumerate(bins):
            regions = _split_regions(placed)
            for (rx, ry, rz, rw, rh, rd) in regions:
                if w <= rw and h <= rh and d <= rd:
                    # candidate: place at rx,ry,rz
                    # score: prefer tight fit (small residual), prefer lower z,y,x
                    residual = (rw - w) * (rh - h) * (rd - d)
                    # also consider placing in corner to maximize future packing
                    score = (rz, ry, rx, residual)
                    if best is None or score < best[0]:
                        best = (score, bi_idx, rx, ry, rz)
        # Try new bin
        if W >= w and H >= h and D >= d:
            score_new = (0, 0, 0, (W - w) * (H - h) * (D - d))
            if best is None or score_new < best[0]:
                best = (score_new, len(bins), 0, 0, 0)
        if best is None:
            return None
        _, bi_idx, px, py, pz = best
        if bi_idx == len(bins):
            bins.append([])
        bins[bi_idx].append((px, py, pz, w, h, d))
        placements.append((bi_idx, px, py, pz, w, h, d))
    return placements


def _score(placements, boxes, W, H, D):
    if placements is None:
        return float('inf')
    max_bin = max(p[0] for p in placements) if placements else 0
    B = max_bin + 1
    last_vol = 0
    for p in placements:
        if p[0] == max_bin:
            last_vol += p[4] * p[5] * p[6]
    cont_vol = W * H * D
    if cont_vol > 0:
        last_frac = last_vol / cont_vol
    else:
        last_frac = 0.0
    return B + 0.1 * (1.0 - last_frac)


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 110.0  # safety margin before 120s kill

    if n == 0:
        return []

    # Generate candidate orderings
    orderings = []

    # 1. Volume descending
    orderings.append(sorted(range(n), key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2])))

    # 2. Max dimension descending
    orderings.append(sorted(range(n), key=lambda i: -max(boxes[i])))

    # 3. Volume / max_dim (tall thin first)
    orderings.append(sorted(range(n), key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2]) / max(1, max(boxes[i]))))

    # 4. Original order
    orderings.append(list(range(n)))

    # 5. Volume ascending (small first to fill gaps)
    orderings.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2]))

    # 6. Max_dim ascending
    orderings.append(sorted(range(n), key=lambda i: max(boxes[i])))

    # Deduplicate orderings
    seen = set()
    unique_orderings = []
    for o in orderings:
        key = tuple(o)
        if key not in seen:
            seen.add(key)
            unique_orderings.append(o)

    best_placements = None
    best_score = float('inf')

    for order in unique_orderings:
        if time.time() > deadline:
            break
        placements = _try_place_order(boxes, order, W, H, D, deadline)
        if placements is not None:
            sc = _score(placements, boxes, W, H, D)
            if sc < best_score:
                best_score = sc
                best_placements = placements

    if best_placements is None:
        # Fallback: simple shelf packing (shouldn't happen but safe)
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements

    return best_placements
# EVOLVE-BLOCK-END
