"""Improved 3D bin packing: max-cubes / best-fit with deadline guard."""
import time
import math

_DEADLINE = 135.0  # safety margin under 150s kill
_START = None

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _fits(bin_boxes, W, H, D, x, y, z, w, h, d):
    if x + w > W or y + h > H or z + d > D:
        return False
    for (bx, by, bz, bw, bh, bd) in bin_boxes:
        # overlap check
        if not (x + w <= bx or bx + bw <= x or
                y + h <= by or by + bh <= y or
                z + d <= bz or bz + bd <= z):
            return False
    return True


def _max_cubes_positions(W, H, D, boxes_in_bin):
    """Generate candidate positions sorted by 'max-cube' potential (corner-first)."""
    xs = [0]
    ys = [0]
    zs = [0]
    for (bx, by, bz, bw, bh, bd) in boxes_in_bin:
        xs.append(bx + bw)
        ys.append(by + bh)
        zs.append(bz + bd)
    xs = sorted(set(v for v in xs if v <= W))
    ys = sorted(set(v for v in ys if v <= H))
    zs = sorted(set(v for v in zs if v <= D))
    # sort by "distance from origin" so corners are tried first (max-cube heuristic)
    cands = []
    for x in xs:
        for y in ys:
            for z in zs:
                cands.append((x, y, z))
    cands.sort(key=lambda p: (-(p[0] + p[1] + p[2]), p[0], p[1], p[2]))
    return cands


def _pack_one(boxes, W, H, D, deadline):
    """Pack boxes into bins; return list of (bin, x, y, z, w, h, d)."""
    bins = []  # each: list of (x,y,z,w,h,d)
    placements = []
    t0 = time.time()

    for (w, h, d) in boxes:
        if time.time() - t0 > deadline:
            break
        best_bin = -1
        best_pos = None
        best_score = None
        for bi, bin_boxes in enumerate(bins):
            cands = _max_cubes_positions(W, H, D, bin_boxes)
            for (x, y, z) in cands:
                if _fits(bin_boxes, W, H, D, x, y, z, w, h, d):
                    # prefer positions that leave large empty space (max-cube)
                    # score: smaller is better; use negative of max-cube volume
                    # approximate max-cube: (W-x-w)*(H-y-h)*(D-z-d)
                    mc = (W - x - w) * (H - y - h) * (D - z - d)
                    score = -mc
                    if best_score is None or score < best_score:
                        best_score = score
                        best_bin = bi
                        best_pos = (x, y, z)
            if best_pos is not None and best_bin == bi:
                # found a fit in this bin; with max-cubes we can stop searching
                # but to be safe, keep checking? No—max-cubes first-fit is standard.
                break
        if best_bin == -1:
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
        placements.append((best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d))
    return placements


def pack(boxes, container):
    W, H, D = container
    deadline = _DEADLINE - _now()
    if deadline <= 0:
        deadline = 5.0
    return _pack_one(boxes, W, H, D, deadline)
