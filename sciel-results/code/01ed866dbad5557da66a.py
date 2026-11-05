"""3D Container Loading: strong Best-Fit-Decreasing with extremal-point search."""
import time
import itertools
import math


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
        if x < px + pw and px < x + w:
            if y < py + ph and py < y + h:
                if z < pz + pd and pz < z + d:
                    return True
    return False


def _fits_inside(x, y, z, w, h, d, W, H, D):
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= W and y + h <= H and z + d <= D)


def _candidate_positions(placed, W, H, D, w, h, d):
    """
    Generate candidate (x,y,z) positions for a box of size (w,h,d).
    Uses extremal points: origin plus cross-product of coordinates from placed boxes,
    clipped so the box fits inside the container.
    """
    cands = set()
    # origin
    if _fits_inside(0, 0, 0, w, h, d, W, H, D):
        cands.add((0, 0, 0))

    # Collect unique x, y, z coordinates from all placed boxes
    xs = set([0])
    ys = set([0])
    zs = set([0])
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px)
        xs.add(px + pw)
        ys.add(py)
        ys.add(py + ph)
        zs.add(pz)
        zs.add(pz + pd)

    # Clip coordinates to valid range
    xs = [x for x in xs if 0 <= x <= W - w]
    ys = [y for y in ys if 0 <= y <= H - h]
    zs = [z for z in zs if 0 <= z <= D - d]

    # Limit candidate set size to avoid explosion on large instances
    if len(xs) > 30:
        xs = sorted(xs)[:30]
    if len(ys) > 30:
        ys = sorted(ys)[:30]
    if len(zs) > 30:
        zs = sorted(zs)[:30]

    # Add all combinations
    for cx in xs:
        for cy in ys:
            for cz in zs:
                if _fits_inside(cx, cy, cz, w, h, d, W, H, D):
                    cands.add((cx, cy, cz))

    return cands


def _softmax(a, b, t=5.0):
    """Smooth approximation of max(a, b) via log-sum-exp."""
    m = max(a, b)
    if m == 0:
        return 0.0
    if a == b:
        return m
    return m + math.log(1.0 + math.exp(-t * abs(a - b))) / t


def _find_best_position(w, h, d, W, H, D, placed):
    """
    Find the best non-overlapping position for a box of size (w,h,d).
    Uses a smooth multi-criteria score. Returns (x,y,z) or None.
    """
    if not placed:
        return (0, 0, 0) if _fits_inside(0, 0, 0, w, h, d, W, H, D) else None

    best = None
    best_score = None
    for (x, y, z) in _candidate_positions(placed, W, H, D, w, h, d):
        if _overlaps(x, y, z, w, h, d, placed):
            continue
        ext_x = x + w
        ext_y = y + h
        footprint_smooth = _softmax(ext_x, ext_y, t=5.0)
        score = (z, footprint_smooth + 0.1 * (x + y) / max(W, H, 1), x, y)
        if best_score is None or score < best_score:
            best_score = score
            best = (x, y, z)
    return best


def _pack_bfd(boxes, W, H, D, deadline, sort_key=None):
    """
    Best Fit Decreasing with orientation search.
    Returns (placements) where placements is a list per original box.
    """
    n = len(boxes)
    if sort_key is None:
        sort_key = lambda i: (-(boxes[i][0] * boxes[i][1] * boxes[i][2]), -max(boxes[i]))
    order = sorted(range(n), key=sort_key)
    placed = []
    result = [None] * n

    for idx in order:
        if _time.time() >= deadline:
            break
        w, h, d = boxes[idx]
        found = False
        best_orient_pos = None
        best_orient_score = None
        for (ow, oh, od) in _orientations(w, h, d):
            pos = _find_best_position(ow, oh, od, W, H, D, placed)
            if pos is not None:
                x, y, z = pos
                ext_x = x + ow
                ext_y = y + oh
                footprint_smooth = _softmax(ext_x, ext_y, t=5.0)
                score = (z, footprint_smooth + 0.1 * (x + y) / max(W, H, 1), x, y)
                if best_orient_score is None or score < best_orient_score:
                    best_orient_score = score
                    best_orient_pos = (x, y, z, ow, oh, od)
        if best_orient_pos is not None:
            x, y, z, ow, oh, od = best_orient_pos
            placed.append((x, y, z, ow, oh, od))
            result[idx] = (0, x, y, z, ow, oh, od)
            found = True
        if not found:
            result[idx] = (-1, 0, 0, 0, w, h, d)

    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    return result


def _pack_shelf(boxes, W, H, D):
    """
    A fast shelf-based fallback.
    Returns per-box placements.
    """
    n = len(boxes)
    order = sorted(range(n), key=lambda i: (-(boxes[i][0] * boxes[i][1] * boxes[i][2]), -max(boxes[i])))
    placed = [None] * n
    x = y = z = 0
    row_h = layer_d = 0
    for idx in order:
        w, h, d = boxes[idx]
        best_orient = None
        best_score = None
        for (ow, oh, od) in _orientations(w, h, d):
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
            if (a[0] < b[0] + b[3] and b[0] < a[0] + a[3] and
                    a[1] < b[1] + b[4] and b[1] < a[1] + a[4] and
                    a[2] < b[2] + b[5] and b[2] < a[2] + a[5]):
                return False
    return True


def _local_search(placements, boxes, W, H, D, deadline):
    """
    Local search: try to relocate boxes to better positions, and try to insert
    boxes that were left out.
    """
    n = len(boxes)
    best = list(placements)
    best_vol = _volume_filled(best, W, H, D)

    # Build placed list from current best
    placed = []
    for i in range(n):
        b, x, y, z, w, h, d = best[i]
        if b == 0:
            placed.append((x, y, z, w, h, d))

    # Try relocating each loaded box to a better position
    for i in range(n):
        if _time.time() >= deadline:
            break
        b, x, y, z, w, h, d = best[i]
        if b != 0:
            continue
        # Remove this box from placed
        old_pos = (x, y, z, w, h, d)
        placed.remove(old_pos)

        # Try to find a better position for this box
        found_improvement = False
        best_new_pos = None
        best_new_score = None

        for (ow, oh, od) in _orientations(w, h, d):
            pos = _find_best_position(ow, oh, od, W, H, D, placed)
            if pos is not None:
                nx, ny, nz = pos
                ext_x = nx + ow
                ext_y = ny + oh
                footprint_smooth = _softmax(ext_x, ext_y, t=5.0)
                score = (nz, footprint_smooth + 0.1 * (nx + ny) / max(W, H, 1), nx, ny)
                # Prefer positions that are lower or more compact
                if best_new_score is None or score < best_new_score:
                    best_new_score = score
                    best_new_pos = (nx, ny, nz, ow, oh, od)

        if best_new_pos is not None:
            nx, ny, nz, ow, oh, od = best_new_pos
            # Check if this is an improvement (lower z or more compact)
            old_ext_x = x + w
            old_ext_y = y + h
            old_footprint = _softmax(old_ext_x, old_ext_y, t=5.0)
            old_score = (z, old_footprint + 0.1 * (x + y) / max(W, H, 1), x, y)
            if best_new_score < old_score:
                placed.append((nx, ny, nz, ow, oh, od))
                best[i] = (0, nx, ny, nz, ow, oh, od)
                found_improvement = True

        if not found_improvement:
            # Put it back in original position
            placed.append(old_pos)

    # Try to insert left-out boxes
    for i in range(n):
        if _time.time() >= deadline:
            break
        b, x, y, z, w, h, d = best[i]
        if b != -1:
            continue
        found = False
        best_orient_pos = None
        best_orient_score = None
        for (ow, oh, od) in _orientations(w, h, d):
            pos = _find_best_position(ow, oh, od, W, H, D, placed)
            if pos is not None:
                nx, ny, nz = pos
                ext_x = nx + ow
                ext_y = ny + oh
                footprint_smooth = _softmax(ext_x, ext_y, t=5.0)
                score = (nz, footprint_smooth + 0.1 * (nx + ny) / max(W, H, 1), nx, ny)
                if best_orient_score is None or score < best_orient_score:
                    best_orient_score = score
                    best_orient_pos = (nx, ny, nz, ow, oh, od)
        if best_orient_pos is not None:
            nx, ny, nz, ow, oh, od = best_orient_pos
            placed.append((nx, ny, nz, ow, oh, od))
            best[i] = (0, nx, ny, nz, ow, oh, od)
            found = True

    return best


def load(boxes, container):
    start = _time.time()
    deadline = start + 140.0  # 10s safety margin under 150s limit

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
    if _time.time() < deadline - 10.0:
        try:
            sort_key2 = lambda i: (-max(boxes[i]), -(boxes[i][0] * boxes[i][1] * boxes[i][2]))
            bfd2 = _pack_bfd(boxes, W, H, D, deadline, sort_key=sort_key2)
            if _validate(bfd2, boxes, W, H, D):
                vol = _volume_filled(bfd2, W, H, D)
                if vol > best_vol:
                    best = bfd2
                    best_vol = vol
        except Exception:
            pass

    # Try local search to improve the best solution found so far
    if _time.time() < deadline - 5.0:
        try:
            improved = _local_search(best, boxes, W, H, D, deadline - 2.0)
            if _validate(improved, boxes, W, H, D):
                vol = _volume_filled(improved, W, H, D)
                if vol > best_vol:
                    best = improved
                    best_vol = vol
        except Exception:
            pass

    # Final safety: ensure validity; if invalid, fall back to shelf.
    if not _validate(best, boxes, W, H, D):
        best = shelf

    return best
# EVOLVE-BLOCK-END
