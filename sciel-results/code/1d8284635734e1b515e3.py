# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(centers, sides, tol=1e-7):
    """Vectorized feasibility check. Returns True if all constraints hold."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol):
        return False
    if np.any(centers[:, 0] + sides / 2 > 1.0 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol):
        return False
    if np.any(centers[:, 1] + sides / 2 > 1.0 + tol):
        return False
    i = np.arange(n)
    j = np.arange(n)
    I, J = np.meshgrid(i, j, indexing='ij')
    mask = I < J
    ii = I[mask]
    jj = J[mask]
    t = (sides[ii] + sides[jj]) / 2.0 - tol
    dx = np.abs(centers[ii, 0] - centers[jj, 0])
    dy = np.abs(centers[ii, 1] - centers[jj, 1])
    if np.any((dx < t) & (dy < t)):
        return False
    return True


def _objective(centers, sides):
    return float(np.sum(sides))


def _max_grow(centers, sides, idx, tol=1e-7):
    """
    For square idx, find the maximum side length s such that we can place
    it (with center adjusted) without overlapping any other square and staying
    in [0,1]^2. Returns (max_side, best_cx, best_cy).
    """
    n = len(sides)
    others = [i for i in range(n) if i != idx]
    s_min = sides[idx]
    s_max = 1.0
    # Binary search on side length
    lo, hi = s_min, s_max
    best_s = s_min
    best_cx = centers[idx, 0]
    best_cy = centers[idx, 1]

    for _ in range(40):
        mid = (lo + hi) / 2.0
        # For this side length, find if there exists a valid center
        # The center must be in [mid/2, 1-mid/2]^2
        # And for each other square j, we need:
        #   |cx - cx_j| >= (mid + s_j)/2 - tol  OR  |cy - cy_j| >= (mid + s_j)/2 - tol
        # This is a 2D feasibility problem. Use a grid search over the valid region.
        found = False
        best_cx_try = centers[idx, 0]
        best_cy_try = centers[idx, 1]

        # Try to place: first try keeping current center, then try corners and edges
        # Actually, let's do a coarse grid search for a valid position
        margin = mid / 2.0
        x_lo, x_hi = margin, 1.0 - margin
        y_lo, y_hi = margin, 1.0 - margin

        if x_lo > x_hi or y_lo > y_hi:
            hi = mid
            continue

        # Coarse grid
        npts = 15
        xs = np.linspace(x_lo, x_hi, npts)
        ys = np.linspace(y_lo, y_hi, npts)
        valid_found = False
        best_x = x_lo
        best_y = y_lo

        for xi in xs:
            for yi in ys:
                ok = True
                for j in others:
                    t = (mid + sides[j]) / 2.0 - tol
                    if abs(xi - centers[j, 0]) < t and abs(yi - centers[j, 1]) < t:
                        ok = False
                        break
                if ok:
                    valid_found = True
                    best_x = xi
                    best_y = yi
                    break
            if valid_found:
                break

        if valid_found:
            lo = mid
            best_s = mid
            best_cx = best_x
            best_cy = best_y
        else:
            hi = mid

    return best_s, best_cx, best_cy


def _greedy_enlarge(centers, sides, tol=1e-7, max_rounds=200, deadline=None):
    """
    Repeatedly try to enlarge each square (in order of current side, largest first)
    until no more growth is possible.
    """
    centers = centers.copy()
    sides = sides.copy()
    n = len(sides)

    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline - 30.0:
            break
        improved = False
        # Process in order of current side (largest first) for better packing
        order = np.argsort(-sides)
        for idx in order:
            if deadline is not None and time.time() > deadline - 30.0:
                break
            new_s, new_cx, new_cy = _max_grow(centers, sides, idx, tol)
            if new_s > sides[idx] + 1e-9:
                sides[idx] = new_s
                centers[idx, 0] = new_cx
                centers[idx, 1] = new_cy
                improved = True
        if not improved:
            break

    return centers, sides


def _corner_push(centers, sides, tol=1e-7):
    """
    Push squares toward corners/edges to free interior space.
    For each square, try moving it to one of the 4 corners or 4 edge midpoints
    (whichever is valid and gives more room for others).
    """
    centers = centers.copy()
    sides = sides.copy()
    n = len(sides)

    # Try pushing each square to nearest corner or edge
    for _ in range(10):
        improved = False
        for i in range(n):
            s = sides[i]
            half = s / 2.0
            # Candidate positions: 4 corners, 4 edge centers, current
            candidates = [
                (half, half), (1 - half, half),
                (half, 1 - half), (1 - half, 1 - half),
                (0.5, half), (0.5, 1 - half),
                (half, 0.5), (1 - half, 0.5),
                (centers[i, 0], centers[i, 1]),
            ]
            for cx, cy in candidates:
                if cx < -tol or cx > 1 + tol or cy < -tol or cy > 1 + tol:
                    continue
                new_c = centers.copy()
                new_c[i] = [cx, cy]
                if _is_valid(new_c, sides, tol):
                    if (cx, cy) != (centers[i, 0], centers[i, 1]):
                        centers[i] = [cx, cy]
                        improved = True
                    break

        if not improved:
            break

    return centers, sides


def _local_refine(centers, sides, tol=1e-7, deadline=None):
    """
    Local refinement: try small random perturbations and accept if valid.
    Also try growing each square.
    """
    centers = centers.copy()
    sides = sides.copy()
    n = len(sides)
    rng = np.random.default_rng(42)
    current_val = _objective(centers, sides)
    best_val = current_val
    best_centers = centers.copy()
    best_sides = sides.copy()

    max_iters = 200000
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 30.0:
            break

        # Random perturbation
        i = rng.integers(n)
        new_c = centers.copy()
        new_s = sides.copy()
        step = 0.02
        new_c[i, 0] += rng.normal(0, step)
        new_c[i, 1] += rng.normal(0, step)
        new_s[i] += rng.normal(0, step * 0.5)

        ns = new_s[i]
        if ns <= 0.001:
            ns = 0.001
        if ns > 1.0:
            ns = 1.0
        new_s[i] = ns
        half = ns / 2.0
        new_c[i, 0] = min(max(new_c[i, 0], half), 1.0 - half)
        new_c[i, 1] = min(max(new_c[i, 1], half), 1.0 - half)

        if _is_valid(new_c, new_s, tol):
            v = _objective(new_c, new_s)
            if v > best_val:
                best_val = v
                best_centers = new_c.copy()
                best_sides = new_s.copy()
                centers = new_c
                sides = new_s

    return best_val, best_centers, best_sides


def _seed_corner_large():
    """
    Layout: 4 large squares in corners + 6 smaller in edges/center.
    Large: side ~0.38, Small: side ~0.24
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    # 4 corner squares
    s_large = 0.38
    centers[0] = [s_large / 2, s_large / 2]
    centers[1] = [1 - s_large / 2, s_large / 2]
    centers[2] = [s_large / 2, 1 - s_large / 2]
    centers[3] = [1 - s_large / 2, 1 - s_large / 2]
    sides[0:4] = s_large

    # 6 smaller squares filling gaps
    s_small = 0.24
    # Top edge middle
    centers[4] = [0.5, 1 - s_small / 2]
    # Bottom edge middle
    centers[5] = [0.5, s_small / 2]
    # Left edge middle
    centers[6] = [s_small / 2, 0.5]
    # Right edge middle
    centers[7] = [1 - s_small / 2, 0.5]
    # Center-ish (two squares)
    centers[8] = [0.42, 0.5]
    centers[9] = [0.58, 0.5]
    sides[4:10] = s_small

    return centers, sides


def _seed_asymmetric():
    """
    One very large square + progressively smaller ones filling the L-shape.
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)

    # Largest square: 0.5 in bottom-left
    s0 = 0.5
    centers[0] = [s0 / 2, s0 / 2]
    sides[0] = s0

    # Fill the remaining L-shaped region
    # Right strip: width 0.5, height 1
    # Top strip: width 0.5, height 0.5 (above the large square)

    # Right side: 2 columns of squares
    s1 = 0.25
    centers[1] = [0.5 + s1 / 2, s1 / 2]
    centers[2] = [0.5 + s1 / 2, 0.5 + s1 / 2]
    centers[3] = [0.5 + s1 / 2, 1.0 - s1 / 2]
    sides[1:4] = s1

    # Far right: 2 columns
    s2 = 0.25
    centers[4] = [1.0 - s2 / 2, s2 / 2]
    centers[5] = [1.0 - s2 / 2, 0.5 + s2 / 2]
    centers[6] = [1.0 - s2 / 2, 1.0 - s2 / 2]
    sides[4:7] = s2

    # Top strip (above large square): 3 squares
    s3 = 1.0 / 6.0
    centers[7] = [s3 / 2, 0.5 + s3 / 2]
    centers[8] = [s3 + s3 / 2, 0.5 + s3 / 2]
    centers[9] = [2 * s3 + s3 / 2, 0.5 + s3 / 2]
    sides[7:10] = s3

    return centers, centers


def _seed_grid_3x3_plus():
    """3x3 grid of side 1/3 + one extra small square."""
    k = 3
    side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(9, side, dtype=np.float64)
    # 10th square: small, in a gap
    centers = np.vstack([centers, np.array([side / 2.0, 1.0 - side / 2.0])])
    sides = np.concatenate([sides, np.array([side])])
    return centers, sides


def _seed_mixed_sizes():
    """
    Mixed sizes: 2 large (0.35), 3 medium (0.25), 5 small (0.18)
    Arranged to fit.
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)

    # 2 large squares
    sL = 0.35
    centers[0] = [sL / 2, sL / 2]
    centers[1] = [1 - sL / 2, sL / 2]
    sides[0:2] = sL

    # 3 medium
    sM = 0.25
    centers[2] = [sM / 2, 1 - sM / 2]
    centers[3] = [0.5, 1 - sM / 2]
    centers[4] = [1 - sM / 2, 1 - sM / 2]
    sides[2:5] = sM

    # 5 small
    sS = 0.18
    # Fill remaining gaps
    pos = [
        [0.5, sL / 2],
        [sL + sS / 2, 0.5 * sL],
        [1 - sL - sS / 2, 0.5 * sL],
        [0.5, 1 - sM - sS / 2],
        [0.25, 0.75],
    ]
    for i, p in enumerate(pos):
        centers[5 + i] = p
        sides[5 + i] = sS

    return centers, sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place 10 axis-aligned squares with
    pairwise disjoint interiors inside the unit square [0,1]^2 so as to
    MAXIMIZE the sum of their side lengths.

    Strategy: deterministic structured packing + greedy iterative enlargement.
    Start from several diverse asymmetric layouts, then use greedy enlargement
    (binary search on each square's side) and corner-pushing to find a good
    packing. Finish with local refinement.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 600.0

    best_val = 0.0
    best_centers = None
    best_sides = None

    # Generate diverse seeds
    seed_funcs = [
        _seed_corner_large,
        _seed_mixed_sizes,
        _seed_grid_3x3_plus,
    ]

    seeds = []
    for sf in seed_funcs:
        try:
            c, s = sf()
            if c.shape == (10, 2) and s.shape == (10,):
                seeds.append((c, s))
        except Exception:
            pass

    # Also add a simple grid as fallback
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    gc = np.stack([cx, cy], axis=1).astype(np.float64)
    gs = np.full(n, side, dtype=np.float64)
    seeds.append((gc, gs))

    for c, s in seeds:
        if time.time() > deadline - 60.0:
            break

        # Ensure feasibility
        if not _is_valid(c, s, tol):
            # Try corner push to fix
            c2, s2 = _corner_push(c, s, tol)
            if _is_valid(c2, s2, tol):
                c, s = c2, s2
            else:
                # Try reducing sides slightly
                s2 = s * 0.9
                c2 = c.copy()
                for i in range(n):
                    half = s2[i] / 2
                    c2[i, 0] = min(max(c2[i, 0], half), 1 - half)
                    c2[i, 1] = min(max(c2[i, 1], half), 1 - half)
                if _is_valid(c2, s2, tol):
                    c, s = c2, s2
                else:
                    # Fallback to grid
                    c, s = gc.copy(), gs.copy()

        if not _is_valid(c, s, tol):
            continue

        # Phase 1: Greedy enlargement
        c, s = _greedy_enlarge(c, s, tol, max_rounds=100, deadline=deadline)

        # Phase 2: Corner push to free space
        c, s = _corner_push(c, s, tol)

        # Phase 3: Another round of greedy enlargement
        c, s = _greedy_enlarge(c, s, tol, max_rounds=100, deadline=deadline)

        # Phase 4: Local refinement
        val, c, s = _local_refine(c, s, tol, deadline=deadline)

        if val > best_val:
            best_val = val
            best_centers = c.copy()
            best_sides = s.copy()

    # Final safety
    if best_centers is None:
        best_centers = gc
        best_sides = gs
    elif not _is_valid(best_centers, best_sides, tol):
        # Try to fix
        c2, s2 = _corner_push(best_centers, best_sides, tol)
        if _is_valid(c2, s2, tol):
            best_centers, best_sides = c2, s2
        else:
            best_centers, best_sides = gc, gs

    return best_centers, best_sides


# EVOLVE-BLOCK-END
