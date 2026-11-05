"""3D Container Loading: strong Best-Fit-Decreasing with extremal-point search."""
import time
import itertools
import heapq

# EVOLVE-BLOCK-START
import time as _time


def _orientations(w, h, d):
    """Yield the 6 distinct axis-permutations of (w, h, d)."""
    seen = set()
    out = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            out.append(perm)
    return out


def _overlaps(x, y, z, w, h, d, placed):
    """Return True if the box at (x,y,z) of size (w,h,d) overlaps any placed box."""
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and px < x + w and
                y < py + ph and py < y + h and
                z < pz + pd and pz < z + d):
            return True
    return False


def _fits_inside(x, y, z, w, h, d, W, H, D):
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= W and y + h <= H and z + d <= D)


def _candidate_positions(placed, W, H, D, w, h, d):
    """
    Generate candidate (x,y,z) positions for a box of size (w,h,d).
    Uses extremal points: origin plus each corner of placed boxes,
    clipped so the box fits inside the container.
    """
    cands = set()
    # origin
    if _fits_inside(0, 0, 0, w, h, d, W, H, D):
        cands.add((0, 0, 0))
    # extremal points from placed boxes
    for (px, py, pz, pw, ph, pd) in placed:
        for cx in (px, px + pw):
            for cy in (py, py + ph):
                for cz in (pz, pz + pd):
                    if _fits_inside(cx, cy, cz, w, h, d, W, H, D):
                        cands.add((cx, cy, cz))
    return cands


def _find_best_position(w, h, d, W, H, D, placed):
    """
    Find the best (lowest, then most -x, then most -y) non-overlapping position
    for a box of size (w,h,d). Returns (x,y,z) or None.
    'Best fit' = minimize z, then x, then y to fill from the bottom.
    """
    best = None
    best_key = None
    for (x, y, z) in _candidate_positions(placed, W, H, D, w, h, d):
        if _overlaps(x, y, z, w, h, d, placed):
            continue
        key = (z, x, y)
        if best_key is None or key < best_key:
            best_key = key
            best = (x, y, z)
    return best


def _pack_bfd(boxes, W, H, D, deadline):
    """
    Best Fit Decreasing with orientation search.
    Returns (placements, placed_list) where placements is a list per original box.
    """
    n = len(boxes)
    # Sort indices by a heuristic: larger volume first, tie-break by max dimension.
    order = sorted(range(n), key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]),
                                            -max(boxes[i])))
    placed = []          # list of (x,y,z,w,h,d)
    result = [None] * n  # per original index: (bin,x,y,z,w,h,d)

    for idx in order:
        if _time.time() >= deadline:
            break
        w, h, d = boxes[idx]
        found = False
        for (ow, oh, od) in _orientations(w, h, d):
            pos = _find_best_position(ow, oh, od, W, H, D, placed)
            if pos is not None:
                x, y, z = pos
                placed.append((x, y, z, ow, oh, od))
                result[idx] = (0, x, y, z, ow, oh, od)
                found = True
                break
        if not found:
            result[idx] = (-1, 0, 0, 0, w, h, d)

    # Fill any that were never processed (deadline hit) as left out.
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    return result


def _pack_shelf(boxes, W, H, D):
    """
    A fast shelf-based fallback (similar to seed but with orientation choice).
    Returns per-box placements.
    """
    n = len(boxes)
    order = sorted(range(n), key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]),
                                            -max(boxes[i])))
    placed = [None] * n
    x = y = z = 0
    row_h = layer_d = 0
    for idx in order:
        w, h, d = boxes[idx]
        # pick the orientation that best fits the current row/layer
        best_orient = None
        best_score = None
        for (ow, oh, od) in _orientations(w, h, d):
            # score: prefer small footprint in x, small height, small depth
            score = (x + ow > W, y + oh > H, z + od > D)
            if not any(score):
                s = (ow, oh, od)
                if best_score is None or s < best_score:
                    best_score = s
                    best_orient = (ow, oh, od)
        if best_orient is None:
            placed[idx] = (-1, 0, 0, 0, w, h, d)
            continue
        ow, oh, od = best_orient
        if x + ow > W:
            x, y, row_h = 0, y + row_h, 0
        if y + oh > H:
            y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
        if z + od > D:
            placed[idx] = (-1, 0, 0, 0, w, h, d)
            continue
        placed[idx] = (0, x, y, z, ow, oh, od)
        x, row_h, layer_d = x + ow, max(row_h, oh), max(layer_d, od)
    return placed


def _volume_filled(placements, W, H, D):
    total = 0
    for (b, x, y, z, w, h, d) in placements:
        if b == 0:
            total += w * h * d
    return total


def _validate(placements, boxes, W, H, D):
    """Check no overlaps and all inside container. Returns True if valid."""
    loaded = []
    for (b, x, y, z, w, h, d) in placements:
        if b != 0:
            continue
        if not _fits_inside(x, y, z, w, h, d, W, H, D):
            return False
        loaded.append((x, y, z, w, h, d))
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            a, b = loaded[i], loaded[j]
            if (a[0] < b[0]+b[3] and b[0] < a[0]+a[3] and
                a[1] < b[1]+b[4] and b[1] < a[1]+a[4] and
                a[2] < b[2]+b[5] and b[2] < a[2]+a[5]):
                return False
    return True


def load(boxes, container):
    start = _time.time()
    deadline = start + 110.0  # 10s safety margin under 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    # Fast shelf fallback first (cheap, always valid).
    shelf = _pack_shelf(boxes, W, H, D)
    best = shelf
    best_vol = _volume_filled(shelf, W, H, D)

    # Try the stronger BFD with extremal-point search.
    try:
        bfd = _pack_bfd(boxes, W, H, D, deadline)
        if _validate(bfd, boxes, W, H, D):
            vol = _volume_filled(bfd, W, H, D)
            if vol > best_vol:
                best = bfd
                best_vol = vol
    except Exception:
        pass

    # If we still have time, try a second BFD pass with a different sort
    # (by max-dimension) to diversify, keeping the better result.
    if _time.time() < deadline - 5.0:
        try:
            boxes2 = list(boxes)
            # Re-run BFD but with a slightly different ordering via a helper
            n = len(boxes)
            order2 = sorted(range(n), key=lambda i: (-max(boxes[i]),
                                                     -(boxes[i][0]*boxes[i][1]*boxes[i][2])))
            placed = []
            result = [None] * n
            for idx in order2:
                if _time.time() >= deadline - 3.0:
                    break
                w, h, d = boxes[idx]
                found = False
                for (ow, oh, od) in _orientations(w, h, d):
                    pos = _find_best_position(ow, oh, od, W, H, D, placed)
                    if pos is not None:
                        x, y, z = pos
                        placed.append((x, y, z, ow, oh, od))
                        result[idx] = (0, x, y, z, ow, oh, od)
                        found = True
                        break
                if not found:
                    result[idx] = (-1, 0, 0, 0, w, h, d)
            for i in range(n):
                if result[i] is None:
                    w, h, d = boxes[i]
                    result[i] = (-1, 0, 0, 0, w, h, d)
            if _validate(result, boxes, W, H, D):
                vol = _volume_filled(result, W, H, D)
                if vol > best_vol:
                    best = result
                    best_vol = vol
        except Exception:
            pass

    # Final safety: ensure validity; if invalid, fall back to shelf.
    if not _validate(best, boxes, W, H, D):
        best = shelf

    return best
# EVOLVE-BLOCK-END
