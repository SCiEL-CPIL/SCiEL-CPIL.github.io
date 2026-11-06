"""Combine grid baseline with vectorized hill-climbing refinement."""
import time
import numpy as np

# Hard wall-clock budget for the whole construct() call (safety margin under 90s).
_DEADLINE = 75.0


def _doubled_areas(pts):
    """Return (areas, tri_idx): areas = |det| for each triple, tri_idx = (i,j,k) arrays.

    Vectorized over all C(n,3) triples. n is small (30) so this is cheap.
    """
    n = len(pts)
    p = np.asarray(pts, dtype=np.int64)
    ix, jx, kx = np.indices((n, n, n))
    # keep strictly i<j<k
    mask = (ix < jx) & (jx < kx)
    i = ix[mask].astype(np.int64)
    j = jx[mask].astype(np.int64)
    k = kx[mask].astype(np.int64)
    xi, yi = p[i, 0], p[i, 1]
    xj, yj = p[j, 0], p[j, 1]
    xk, yk = p[k, 0], p[k, 1]
    det = (xi * (yj - yk)) + (xj * (yk - yi)) + (xk * (yi - yj))
    areas = np.abs(det)
    return areas, (i, j, k)


def _best_multiplicity(areas):
    """Largest multiplicity of a single positive doubled area."""
    pos = areas[areas > 0]
    if pos.size == 0:
        return 0
    vals, counts = np.unique(pos, return_counts=True)
    return int(counts.max())


def _evaluate(pts):
    areas, _ = _doubled_areas(pts)
    return _best_multiplicity(areas)


def _hill_climb(pts, deadline):
    """Local search: move a point to a nearby empty integer cell if it improves
    the max positive-area multiplicity (tie-break: total area concentration)."""
    pts = [tuple(p) for p in pts]
    cur = _evaluate(pts)
    cur_set = set(pts)

    # Precompute all candidate cells in range.
    cells = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each point; for each, try a bounded set of nearby moves.
        for idx in range(len(pts)):
            if time.time() >= deadline:
                break
            ox, oy = pts[idx]
            # candidate new positions: neighbors within radius 2 (and a few jumps)
            cands = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in cur_set:
                            cands.append((nx, ny))
            # Also try a few random distant cells to escape local minima.
            import random
            for _ in range(4):
                c = random.choice(cells)
                if c not in cur_set:
                    cands.append(c)
            for c in cands:
                if time.time() >= deadline:
                    break
                new_pts = list(pts)
                new_pts[idx] = c
                new_set = cur_set - {pts[idx]} | {c}
                # quick filter: must keep 30 distinct
                if len(new_set) != 30:
                    continue
                new_val = _evaluate(new_pts)
                if new_val > cur:
                    # accept
                    cur_set = new_set
                    pts = new_pts
                    cur = new_val
                    improved = True
                    break
    return pts, cur


def construct():
    start = time.time()
    deadline = start + _DEADLINE

    # Baseline: 6x5 grid (the seed) centered to use range well.
    base = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    best_pts = [tuple(p) for p in base]
    best_val = _evaluate(best_pts)

    # Hill-climb from the baseline.
    pts, val = _hill_climb(best_pts, deadline)
    if val > best_val:
        best_pts, best_val = pts, val

    # If time remains, try a couple of alternative seeds and hill-climb each.
    if time.time() < deadline - 5:
        # Seed 2: two long parallel lines (from inspiration) -- good for small-area multiplicity.
        seed2 = [(x, 0) for x in range(-7, 8)] + [(x, 1) for x in range(-7, 8)]
        p2, v2 = _hill_climb(seed2, deadline)
        if v2 > best_val:
            best_pts, best_val = p2, v2

    # Final guard: ensure exactly 30 distinct points in range.
    seen = set()
    out = []
    for p in best_pts:
        x, y = p
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
    # If somehow short (shouldn't happen), pad with fresh cells.
    if len(out) < 30:
        for c in [(x, y) for x in range(-30, 31) for y in range(-30, 31)]:
            if len(out) >= 30:
                break
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out[:30]
