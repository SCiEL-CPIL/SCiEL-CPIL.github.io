"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return unique orientations of a box (w, h, d)."""
    seen = set()
    result = []
    for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(px, py, pz, w, h, d, occupied, W, H, D):
    """Check if box at (px,py,pz) with dims (w,h,d) fits without overlap."""
    if px + w > W or py + h > H or pz + d > D:
        return False
    for (ox, oy, oz, ow, oh, od) in occupied:
        if (px < ox + ow and px + w > ox and
            py < oy + oh and py + h > oy and
            pz < oz + od and pz + d > oz):
            return False
    return True

def _extreme_points(occupied, W, H, D):
    """Generate candidate positions from faces/corners of placed boxes + origin."""
    pts = set()
    pts.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in occupied:
        # Six face-adjacent positions
        pts.add((px + pw, py, pz))
        pts.add((px, py + ph, pz))
        pts.add((px, py, pz + pd))
        # Three edge-adjacent positions
        pts.add((px + pw, py + ph, pz))
        pts.add((px + pw, py, pz + pd))
        pts.add((px, py + ph, pz + pd))
        # One corner position
        pts.add((px + pw, py + ph, pz + pd))
    # Filter to valid positions
    valid = []
    for (x, y, z) in pts:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            valid.append((x, y, z))
    return valid

def _score_position(px, py, pz, tw, th, td, W, H, D):
    """Score a placement: prefer positions that minimize wasted space in each axis."""
    # Prefer lower z (gravity), then lower y, then lower x
    # Also prefer orientations that align with container dimensions
    score = 0
    score += (W - px - tw) * 0.1  # leftover x space
    score += (H - py - th) * 0.1  # leftover y space
    score += (D - pz - td) * 1.0  # leftover z space (most important - gravity)
    return score

def _greedy(order, boxes, container):
    """Greedy placement using extreme points."""
    W, H, D = container
    n = len(boxes)
    occupied = []
    result = [None] * n
    used_volume = 0.0

    for idx in order:
        w, h, d = boxes[idx]
        best_pos = None
        best_orient = None
        best_score = float('inf')

        orientations = _rotations(w, h, d)
        candidates = _extreme_points(occupied, W, H, D)

        for (tw, th, td) in orientations:
            for (px, py, pz) in candidates:
                if _fits(px, py, pz, tw, th, td, occupied, W, H, D):
                    s = _score_position(px, py, pz, tw, th, td, W, H, D)
                    if s < best_score:
                        best_score = s
                        best_pos = (px, py, pz)
                        best_orient = (tw, th, td)

        if best_pos is not None:
            px, py, pz = best_pos
            tw, th, td = best_orient
            occupied.append((px, py, pz, tw, th, td))
            result[idx] = (0, px, py, pz, tw, th, td)
            used_volume += tw * th * td

    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)

    return result, used_volume

def _local_search(result, boxes, container, start_time, deadline):
    """Try to improve by moving unplaced boxes or repositioning placed ones."""
    W, H, D = container
    n = len(boxes)
    occupied = []
    for i in range(n):
        if result[i][0] == 0:
            _, px, py, pz, tw, th, td = result[i]
            occupied.append((px, py, pz, tw, th, td))

    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        # Try to place each unplaced box
        for i in range(n):
            if result[i][0] == -1:
                w, h, d = boxes[i]
                orientations = _rotations(w, h, d)
                candidates = _extreme_points(occupied, W, H, D)
                best_pos = None
                best_orient = None
                best_score = float('inf')

                for (tw, th, td) in orientations:
                    for (px, py, pz) in candidates:
                        if _fits(px, py, pz, tw, th, td, occupied, W, H, D):
                            s = _score_position(px, py, pz, tw, th, td, W, H, D)
                            if s < best_score:
                                best_score = s
                                best_pos = (px, py, pz)
                                best_orient = (tw, th, td)

                if best_pos is not None:
                    px, py, pz = best_pos
                    tw, th, td = best_orient
                    result[i] = (0, px, py, pz, tw, th, td)
                    occupied.append((px, py, pz, tw, th, td))
                    improved = True

    return result

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0
    W, H, D = container
    n = len(boxes)

    # Multiple heuristic orderings
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    max_dim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    aspect_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] / max(boxes[i]), reverse=True)
    area_order = sorted(range(n), key=lambda i: max(boxes[i][0]*boxes[i][1], boxes[i][0]*boxes[i][2], boxes[i][1]*boxes[i][2]), reverse=True)

    best_result = None
    best_volume = 0

    orderings = [vol_order, max_dim_order, aspect_order, area_order]

    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        result, vol = _greedy(order, boxes, container)
        if vol > best_volume:
            best_volume = vol
            best_result = result[:]

    if best_result is None:
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    # Local search to try to place more boxes
    best_result = _local_search(best_result, boxes, container, start_time, DEADLINE)

    return best_result

# EVOLVE-BLOCK-END
