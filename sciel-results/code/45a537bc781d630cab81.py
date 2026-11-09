# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 570.0          # 600 s wall limit minus 30 s safety margin
_TOL = 1e-7


def _time_ok():
    return time.time() < _DEADLINE


def _feasible(centers, sides):
    """Return True if all squares are inside [0,1]^2 (with tol) and pairwise
    interiors are disjoint (touching allowed)."""
    n = len(sides)
    c = np.asarray(centers, dtype=float)
    s = np.asarray(sides, dtype=float)
    if np.any(~np.isfinite(c)) or np.any(~np.isfinite(s)):
        return False
    if np.any(s <= 0.0):
        return False
    # inside unit square
    if np.any(c[0] - s / 2.0 < -_TOL) or np.any(c[0] + s / 2.0 > 1.0 + _TOL):
        return False
    if np.any(c[1] - s / 2.0 < -_TOL) or np.any(c[1] + s / 2.0 > 1.0 + _TOL):
        return False
    # pairwise interior-disjoint
    for i in range(n):
        for j in range(i + 1, n):
            t = (s[i] + s[j]) / 2.0 - _TOL
            if abs(c[i, 0] - c[j, 0]) < t and abs(c[i, 1] - c[j, 1]) < t:
                return False
    return True


def _score(centers, sides):
    return float(np.sum(sides))


def _grid_5x2():
    """5 columns x 2 rows, all side 0.2. Sum = 2.0."""
    centers = []
    sides = []
    for r in range(2):
        for cc in range(5):
            centers.append(((cc + 0.5) * 0.2, (r + 0.5) * 0.2))
            sides.append(0.2)
    return np.array(centers), np.array(sides)


def _grid_3x3_plus1():
    """3x3 grid of side 1/3 in bottom 2/3, plus one square in the top band.
    Sum = 3.0 + 1/3 = 3.333."""
    centers = []
    sides = []
    s = 1.0 / 3.0
    for r in range(3):
        for cc in range(3):
            centers.append(((cc + 0.5) * s, (r + 0.5) * s))
            sides.append(s)
    # top band: y in [2/3, 1], height 1/3 -> square of side 1/3
    centers.append((0.5, 2.0 / 3.0 + 0.5 / 3.0))
    sides.append(1.0 / 3.0)
    return np.array(centers), np.array(sides)


def _grid_2x5():
    """2 columns x 5 rows, all side 0.5. Sum = 5.0 (same as 5x2, transposed)."""
    centers = []
    sides = []
    for r in range(5):
        for cc in range(2):
            centers.append(((cc + 0.5) * 0.5, (r + 0.5) * 0.5))
            sides.append(0.5)
    return np.array(centers), np.array(sides)


def _row_10():
    """10 squares in a row, side 0.1. Sum = 1.0 (weak, but valid)."""
    centers = []
    sides = []
    for i in range(10):
        centers.append(((i + 0.5) * 0.1, 0.5))
        sides.append(0.1)
    return np.array(centers), np.array(sides)


def _big_square_packing(bx, by, bs, rng):
    """Place a big square (side bs) at (bx,by). Then greedily place the other 9
    squares in the free space to the right (x>bs) and above (y>bs). Returns
    (centers, sides, valid)."""
    centers = [(bx, by)]
    sides = [bs]
    # free rectangles (x0, x1, y0, y1)
    rects = []
    if bs < 1.0 - 1e-9:
        rects.append((bs, 1.0, 0.0, 1.0))          # right strip
        rects.append((0.0, 1.0, bs, 1.0))          # top strip
    for _ in range(9):
        best = None  # (side, rect_index, cx, cy)
        for ri, (x0, x1, y0, y1) in enumerate(rects):
            w = x1 - x0
            h = y1 - y0
            if w <= 1e-9 or h <= 1e-9:
                continue
            side = min(w, h)
            if side <= 1e-9:
                continue
            # place at the lower-left of this rectangle (deterministic)
            cx = x0 + side / 2.0
            cy = y0 + side / 2.0
            # add a small jitter to break symmetry / explore
            if best is None or side > best[0] + 1e-12:
                best = (side, ri, cx, cy)
        if best is None:
            return None, None, False
        side, ri, cx, cy = best
        x0, x1, y0, y1 = rects[ri]
        centers.append((cx, cy))
        sides.append(side)
        # split the rectangle around the placed square
        # placed square occupies [cx-side/2, cx+side/2] x [cy-side/2, cy+side/2]
        # which equals [x0, x0+side] x [y0, y0+side]
        px0, px1 = x0, x0 + side
        py0, py1 = y0, y0 + side
        new_rects = []
        # right part of this rect
        if x1 - px1 > 1e-9:
            new_rects.append((px1, x1, y0, y1))
        # top part of this rect
        if y1 - py1 > 1e-9:
            new_rects.append((px0, px1, py1, y1))
        # replace rect ri with new_rects
        rects[ri] = None
        rects = [r for r in rects if r is not None]
        rects.extend(new_rects)
    centers = np.array(centers)
    sides = np.array(sides)
    if not _feasible(centers, sides):
        return None, None, False
    return centers, sides, True


def _optimize_big_square(rng, iters=40):
    """Local search over (bx, by, bs) using the big-square packing. Returns
    best (centers, sides, score) found, or None."""
    best = None
    for it in range(iters):
        if not _time_ok():
            break
        # random initial big square
        bs = rng.uniform(0.35, 0.65)
        bx = rng.uniform(0.0, 1.0 - bs)
        by = rng.uniform(0.0, 1.0 - bs)
        c, s, ok = _big_square_packing(bx, by, bs, rng)
        if not ok:
            continue
        sc = _score(c, s)
        if best is None or sc > best[2]:
            best = (c, s, sc)
        # local refinement with Nelder-Mead on a smoothed objective
        def neg_sum(v):
            bxv, byv, bsv = v
            bsv = min(max(bsv, 0.1), 0.8)
            bxv = min(max(bxv, 0.0), 1.0 - bsv)
            byv = min(max(byv, 0.0), 1.0 - bsv)
            cc, ss, o = _big_square_packing(bxv, byv, bsv, rng)
            if not o:
                return -1.0  # infeasible -> very negative
            return -_score(cc, ss)
        x0 = np.array([bx, by, bs])
        try:
            res = minimize(neg_sum, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5,
                                    'adaptive': True})
        except Exception:
            res = None
        if res is not None and res.x is not None:
            bxv, byv, bsv = res.x
            bsv = min(max(bsv, 0.1), 0.8)
            bxv = min(max(bxv, 0.0), 1.0 - bsv)
            byv = min(max(byv, 0.0), 1.0 - bsv)
            cc, ss, o = _big_square_packing(bxv, byv, bsv, rng)
            if o:
                sc2 = _score(cc, ss)
                if best is None or sc2 > best[2]:
                    best = (cc, ss, sc2)
    return best


def squares_in_square10() -> tuple:
    """Maximize the sum of side lengths of 10 axis-aligned, interior-disjoint
    squares inside the unit square. Returns (centers, sides) with centers of
    shape (10,2) and sides of shape (10,)."""
    best = None
    # 1) deterministic packings (instant, no risk)
    for fn in (_grid_5x2, _grid_3x3_plus1, _grid_2x5, _row_10):
        c, s = fn()
        if _feasible(c, s):
            sc = _score(c, s)
            if best is None or sc > best[2]:
                best = (c, s, sc)
    # 2) local search over big-square packings
    rng = np.random.default_rng(12345)
    it = 0
    while _time_ok() and it < 2000:
        it += 1
        r = _optimize_big_square(rng, iters=30)
        if r is not None and (best is None or r[2] > best[2]):
            best = r
    # 3) final safety: if something went wrong, fall back to a known-good grid
    if best is None or not _feasible(best[0], best[1]):
        c, s = _grid_5x2()
        best = (c, s, _score(c, s))
    return best[0].astype(np.float64), best[1].astype(np.float64)


# EVOLVE-BLOCK-END
