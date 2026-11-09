# EVOLVE-BLOCK-START
import numpy as np
import time


def _validate(centers, sides, tol=1e-7):
    """Return True if the packing is feasible (inside box, no overlap)."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if not np.all(sides > 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1.0 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t <= 0:
                continue
            if abs(centers[i, 0] - centers[j, 0]) < t and \
               abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _free_range(centers, sides, i, tol=1e-7):
    """Maximal free interval along an axis for square i given the others."""
    cx, cy = centers[i]
    s = sides[i]
    xlo, xhi = 0.0, 1.0
    ylo, yhi = 0.0, 1.0
    for j in range(len(sides)):
        if j == i:
            continue
        s2 = sides[j]
        dx = (s + s2) / 2.0 + tol
        dy = (s + s2) / 2.0 + tol
        # x-overlap?
        if abs(cx - centers[j, 0]) < dx:
            # y must separate: push i's x away from j's x
            xlo = max(xlo, centers[j, 0] + dx - s / 2.0)
            xhi = min(xhi, centers[j, 0] - dx + s / 2.0)
        if abs(cy - centers[j, 1]) < dy:
            ylo = max(ylo, centers[j, 1] + dy - s / 2.0)
            yhi = min(yhi, centers[j, 1] - dy + s / 2.0)
    return (xlo, xhi), (ylo, yhi)


def _grow_into_gaps(centers, sides, tol=1e-7, max_iters=2000, step=0.02):
    """Grow each square toward its free-space center until no more gain."""
    centers = centers.copy().astype(np.float64)
    sides = sides.copy().astype(np.float64)
    n = len(sides)
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(n):
            (xlo, xhi), (ylo, yhi) = _free_range(centers, sides, i, tol)
            if xhi < xlo or yhi < ylo:
                continue
            # free width/height available for this square
            fx = xhi - xlo
            fy = yhi - ylo
            # max side we could reach
            new_s = max(sides[i], min(fx, fy))
            if new_s <= sides[i] + 1e-9:
                continue
            # move toward free center to maximize room
            nx = 0.5 * (xlo + xhi)
            ny = 0.5 * (ylo + yhi)
            centers[i, 0] = nx
            centers[i, 1] = ny
            sides[i] = new_s
            improved = True
    return centers, sides


def _ascent(centers, sides, tol=1e-7, max_iters=4000, step=0.02):
    """Coordinate ascent: nudge centers / grow sides to increase total."""
    centers = centers.copy().astype(np.float64)
    sides = sides.copy().astype(np.float64)
    n = len(sides)
    best = float(np.sum(sides))
    for it in range(max_iters):
        improved = False
        # try growing each square within its free box
        for i in range(n):
            (xlo, xhi), (ylo, yhi) = _free_range(centers, sides, i, tol)
            if xhi < xlo or yhi < ylo:
                continue
            new_s = min(xhi - xlo, yhi - ylo)
            if new_s > sides[i] + 1e-9:
                centers[i, 0] = 0.5 * (xlo + xhi)
                centers[i, 1] = 0.5 * (ylo + yhi)
                sides[i] = new_s
                improved = True
        # small center nudges to unlock more room
        for i in range(n):
            for dx in (-step, step):
                for dy in (-step, step):
                    trial_c = centers.copy()
                    trial_c[i, 0] += dx
                    trial_c[i, 1] += dy
                    if _validate(trial_c, sides, tol):
                        # does this allow a bigger square i?
                        (xlo, xhi), (ylo, yhi) = _free_range(trial_c, sides, i, tol)
                        if xhi >= xlo and yhi >= ylo:
                            new_s = min(xhi - xlo, yhi - ylo)
                            if new_s > sides[i] + 1e-9:
                                centers[i, 0] = 0.5 * (xlo + xhi)
                                centers[i, 1] = 0.5 * (ylo + yhi)
                                sides[i] = new_s
                                improved = True
                                break
                if improved:
                    break
        cur = float(np.sum(sides))
        if cur > best:
            best = cur
        if not improved:
            break
    return centers, sides


def _tiling_2plus8():
    """One big square (side 1/2) top-left, eight small (side 1/4) in the rest."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    # Big square: occupies [0,0.5]x[0.5,1]
    centers[0] = [0.25, 0.75]
    sides[0] = 0.5
    # Small squares side 0.25. Place them in:
    #  bottom strip y in [0,0.5]: x columns 0,0.25,0.5,0.75 -> 4 squares
    #  right strip x in [0.5,1], y in [0.5,1]: x 0.625,0.875? need 4 more
    #  Actually right strip is 0.5 wide x 0.5 tall -> 4 squares of 0.25
    idx = 1
    # bottom strip: y center 0.125 and 0.375; x centers 0.125,0.375,0.625,0.875
    for yc in (0.125, 0.375):
        for xc in (0.125, 0.375):
            centers[idx] = [xc, yc]
            sides[idx] = 0.25
            idx += 1
    # right strip: x centers 0.625,0.875; y centers 0.625,0.875
    for xc in (0.625, 0.875):
        for yc in (0.625, 0.875):
            centers[idx] = [xc, yc]
            sides[idx] = 0.25
            idx += 1
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in the unit square.

    Strategy: start from a 2+8 tiling (sum=3.0) and use free-space coordinate
    ascent to grow squares into the gaps the tiling leaves.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # safety margin

    centers, sides = _tiling_2plus8()
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = float(np.sum(sides))

    # Phase 1: grow into gaps
    c, s = _grow_into_gaps(centers, sides)
    if _validate(c, s):
        val = float(np.sum(s))
        if val > best_val:
            best_centers, best_sides, best_val = c.copy(), s.copy(), val

    # Phase 2: coordinate ascent with nudges
    if time.time() < deadline:
        c, s = _ascent(best_centers, best_sides)
        if _validate(c, s):
            val = float(np.sum(s))
            if val > best_val:
                best_centers, best_sides, best_val = c.copy(), s.copy(), val

    # Phase 3: a few local restarts with small perturbations
    if time.time() < deadline:
        rng = np.random.default_rng(12345)
        for _ in range(20):
            if time.time() >= deadline:
                break
            c = best_centers + rng.normal(0, 0.01, size=best_centers.shape)
            s = best_sides * (1.0 + rng.normal(0, 0.02, size=best_sides.shape))
            s = np.clip(s, 1e-4, 1.0)
            c, s = _grow_into_gaps(c, s)
            if _validate(c, s):
                val = float(np.sum(s))
                if val > best_val:
                    best_centers, best_sides, best_val = c.copy(), s.copy(), val
                    c, s = _ascent(best_centers, best_sides)
                    if _validate(c, s):
                        val = float(np.sum(s))
                        if val > best_val:
                            best_centers, best_sides, best_val = c.copy(), s.copy(), val

    # Ensure final result is valid; fall back to seed grid if not
    if not _validate(best_centers, best_sides):
        n = 10
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers, best_sides


# EVOLVE-BLOCK-END
