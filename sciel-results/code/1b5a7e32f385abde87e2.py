"""3D Container Loading: BFD with smooth scoring, local relocation, and gap-filling insertion."""
import time
import itertools
import math
import random

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
    Uses extremal points, floor projections, and wall projections for gap-filling.
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

    # Limit candidate set size to avoid explosion
    if len(xs) > 30:
        xs = sorted(xs)[:30]
    if len(ys) > 30:
        ys = sorted(ys)[:30]
    if len(zs) > 30:
        zs = sorted(zs)[:30]

    # Full cross-product of coordinates
    for cx in xs:
        for cy in ys:
            for cz in zs:
                if _fits_inside(cx, cy, cz, w, h, d, W, H, D):
                    cands.add((cx, cy, cz))

    # Floor projections: for each (x,y) from placed boxes, try z=0
    for (px, py, pz, pw, ph, pd) in placed:
        for cx in (px, px + pw):
            for cy in (py, py + ph):
                if 0 <= cx <= W - w and 0 <= cy <= H - h:
                    if _fits_inside(cx, cy, 0, w, h, d, W, H, D):
                        cands.add((cx, cy, 0))

    # Wall projections: try x=0 and y=0 planes
    for (px, py, pz, pw, ph, pd) in placed:
        for cz in (pz, pz + pd):
            if 0 <= cz <= D - d:
                if _fits_inside(0, py, cz, w, h, d, W, H, D):
                    cands.add((0, py, cz))
                if _fits_inside(0, py + ph, cz, w, h, d, W, H, D):
                    cands.add((0, py + ph, cz))
                if _fits_inside(px, 0, cz, w, h, d, W, H, D):
                    cands.add((px, 0, cz))
                if _fits_inside(px + pw, 0, cz, w, h, d, W, H, D):
                    cands.add((px + pw, 0, cz))

    return cands


def _softmax(a, b, t=5.0):
    """Smooth approximation of max(a, b) via log-sum-exp."""
    m = max(a, b)
    if m == 0:
        return 0.0
    if a == b:
        return m
    return m + math.log(1.0 + math.exp(-t * abs(a - b))) / t


def _position_score(x, y, z, w, h, W, H):
    """Score a position: lower is better. Primarily minimize z, then smooth footprint."""
    ext_x = x + w
    ext_y = y + h
    footprint_smooth = _softmax(ext_x, ext_y, t=5.0)
    # Small penalty for large x+y to prefer lower-left corner
    penalty = 0.1 * (x + y) / max(W, H, 1)
    return (z, footprint_smooth + penalty, x, y)


def _find_best_position(w, h, d, W, H, D, placed):
    """
    Find the best non-overlapping position for a box of size (w,h,d).
    Uses smooth multi-criteria scoring. Returns (x,y,z) or None.
    """
    if not placed:
        if _fits_inside(0, 0, 0, w, h, d, W, H, D):
            return (0, 0, 0)
        return None

    best = None
    best_score = None
    for (x, y, z) in _candidate_positions(placed, W, H, D, w, h, d):
        if _overlaps(x, y, z, w, h, d, placed):
            continue
        score = _position_score(x, y, z, w, h, W, H)
        if best_score is None or score < best_score:
            best_score = score
            best = (x, y, z)
    return best


def _pack_bfd(boxes, W, H, D, deadline, sort_key=None):
    """
    Best Fit Decreasing with orientation search.
    Returns per-box placements list.
    """
    n = len(boxes)
    if sort_key is None:
        sort_key = lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]), -max(boxes[i]))
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
                score = _position_score(x, y, z, ow, oh, W, H)
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
    """Fast shelf-based fallback."""
    n = len(boxes)
    order = sorted(range(n), key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]), -max(boxes[i])))
    placed = [None] * n
    x = y = z = 0
    row_h = layer_d = 0
    for idx in order:
        w, h, d = boxes[idx]
        best_orient = None
        best_score = None
        for (ow, oh, od) in _orientations(w, h, d):
            if x + ow <= W and y + oh <= H and z + od <= D:
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


def _volume_filled(placements):
    total = 0
    for (b, x, y, z, w, h, d) in placements:
        if b == 0:
            total += w * h * d
    return total


def _validate(placements, W, H, D):
    """Check no overlaps and all inside container."""
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


def _local_improve(result, boxes, W, H, D, deadline, max_passes=3):
    """
    Local search: try relocating placed boxes to better positions and
    inserting left-out boxes into freed gaps.
    """
    n = len(boxes)
    # Build placed list from current result
    placed_idx = []  # (box_index, x, y, z, w, h, d)
    for i in range(n):
        b, x, y, z, w, h, d = result[i]
        if b == 0:
            placed_idx.append((i, x, y, z, w, h, d))

    # Try to relocate each placed box to a better position
    improved = False
    for pass_num in range(max_passes):
        if _time.time() >= deadline:
            break
        for pi in range(len(placed_idx)):
            if _time.time() >= deadline:
                break
            box_i, ox, oy, oz, ow, oh, od = placed_idx[pi]
            # Remove this box from placed list
            remaining = [p for j, p in enumerate(placed_idx) if j != pi]
            # Try to find a better position for this box
            best_pos = None
            best_score = None
            for (rw, rh, rd) in _orientations(ow, oh, od):
                pos = _find_best_position(rw, rh, rd, W, H, D, remaining)
                if pos is not None:
                    x, y, z = pos
                    score = _position_score(x, y, z, rw, rh, W, H)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (x, y, z, rw, rh, rd)
            if best_pos is not None:
                x, y, z, rw, rh, rd = best_pos
                # Only accept if it's actually better (lower z or tighter footprint)
                old_score = _position_score(ox, oy, oz, ow, oh, W, H)
                if best_score < old_score:
                    placed_idx[pi] = (box_i, x, y, z, rw, rh, rd)
                    result[box_i] = (0, x, y, z, rw, rh, rd)
                    improved = True

    # Try to insert left-out boxes into gaps
    for i in range(n):
        if _time.time() >= deadline:
            break
        b, x, y, z, w, h, d = result[i]
        if b != -1:
            continue
        # Try to place this box
        remaining = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placed_idx]
        best_orient_pos = None
        best_orient_score = None
        for (ow, oh, od) in _orientations(w, h, d):
            pos = _find_best_position(ow, oh, od, W, H, D, remaining)
            if pos is not None:
                px, py, pz = pos
                score = _position_score(px, py, pz, ow, oh, W, H)
                if best_orient_score is None or score < best_orient_score:
                    best_orient_score = score
                    best_orient_pos = (px, py, pz, ow, oh, od)
        if best_orient_pos is not None:
            px, py, pz, ow, oh, od = best_orient_pos
            # Verify no overlap with all placed
            all_placed = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placed_idx]
            if not _overlaps(px, py, pz, ow, oh, od, all_placed):
                result[i] = (0, px, py, pz, ow, oh, od)
                placed_idx.append((i, px, py, pz, ow, oh, od))
                improved = True

    return result, improved


def _randomized_bfd(boxes, W, H, D, deadline, seed=0):
    """
    Randomized BFD: perturb the sort order and candidate selection slightly.
    Returns per-box placements list.
    """
    rng = random.Random(seed)
    n = len(boxes)
    
    # Create a perturbed sort key
    base_key = lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]), -max(boxes[i]))
    perturbed_order = list(range(n))
    # Randomly swap some adjacent elements
    for _ in range(min(10, n // 3)):
        if len(perturbed_order) > 1:
            i = rng.randint(0, len(perturbed_order) - 2)
            perturbed_order[i], perturbed_order[i+1] = perturbed_order[i+1], perturbed_order[i]
    
    # Sort by base key, but with random tie-breaking
    def sort_key(i):
        base = base_key(i)
        # Add small random perturbation to break ties
        return (base[0] + rng.random() * 0.001, base[1])
    
    order = sorted(perturbed_order, key=sort_key)
    
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
                score = _position_score(x, y, z, ow, oh, W, H)
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


def load(boxes, container):
    start = _time.time()
    deadline = start + 140.0  # 10s safety margin under 150s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    # Fast shelf fallback first (cheap, always valid).
    shelf = _pack_shelf(boxes, W, H, D)
    best = shelf
    best_vol = _volume_filled(shelf)

    # Primary BFD with volume sort
    try:
        bfd = _pack_bfd(boxes, W, H, D, deadline)
        if _validate(bfd, W, H, D):
            vol = _volume_filled(bfd)
            if vol > best_vol:
                best = bfd
                best_vol = vol
    except Exception:
        pass

    # Local improvement on the best result
    if _time.time() < deadline - 15.0:
        try:
            improved, _ = _local_improve(best, boxes, W, H, D, deadline - 10.0, max_passes=3)
            if _validate(improved, W, H, D):
                vol = _volume_filled(improved)
                if vol > best_vol:
                    best = improved
                    best_vol = vol
        except Exception:
            pass

    # Second BFD with different sort (by max dimension) for diversity
    if _time.time() < deadline - 20.0:
        try:
            n = len(boxes)
            sort_key2 = lambda i: (-max(boxes[i]), -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
            bfd2 = _pack_bfd(boxes, W, H, D, deadline - 15.0, sort_key=sort_key2)
            if _validate(bfd2, W, H, D):
                vol = _volume_filled(bfd2)
                if vol > best_vol:
                    best = bfd2
                    best_vol = vol
                    # Local improve this one too
                    if _time.time() < deadline - 10.0:
                        try:
                            improved2, _ = _local_improve(best, boxes, W, H, D, deadline - 5.0, max_passes=2)
                            if _validate(improved2, W, H, D):
                                vol2 = _volume_filled(improved2)
                                if vol2 > best_vol:
                                    best = improved2
                                    best_vol = vol2
                        except Exception:
                            pass
        except Exception:
            pass

    # Third BFD with a third sort (by min dimension, to fill tight spaces)
    if _time.time() < deadline - 15.0:
        try:
            n = len(boxes)
            sort_key3 = lambda i: (min(boxes[i]), -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
            bfd3 = _pack_bfd(boxes, W, H, D, deadline - 10.0, sort_key=sort_key3)
            if _validate(bfd3, W, H, D):
                vol = _volume_filled(bfd3)
                if vol > best_vol:
                    best = bfd3
                    best_vol = vol
        except Exception:
            pass

    # Randomized BFD multi-start for additional diversity
    if _time.time() < deadline - 25.0:
        try:
            for seed in range(5):
                if _time.time() >= deadline - 20.0:
                    break
                rand_bfd = _randomized_bfd(boxes, W, H, D, deadline - 20.0 + seed * 3, seed=seed)
                if _validate(rand_bfd, W, H, D):
                    vol = _volume_filled(rand_bfd)
                    if vol > best_vol:
                        best = rand_bfd
                        best_vol = vol
                        # Quick local improve
                        if _time.time() < deadline - 15.0:
                            try:
                                improved_r, _ = _local_improve(best, boxes, W, H, D, deadline - 10.0, max_passes=2)
                                if _validate(improved_r, W, H, D):
                                    vol_r = _volume_filled(improved_r)
                                    if vol_r > best_vol:
                                        best = improved_r
                                        best_vol = vol_r
                            except Exception:
                                pass
        except Exception:
            pass

    # Final safety: ensure validity
    if not _validate(best, W, H, D):
        best = shelf

    return best
# EVOLVE-BLOCK-END
