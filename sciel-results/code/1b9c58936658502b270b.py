"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # distance to each of the 4 walls
    wall = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5),
    )
    wall_min = wall.min()
    # nearest neighbour distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    return min(wall_min, 0.5 * nn)


def _make_hex(rows, spacing, dy, xoff):
    """Build a hex lattice with the given row lengths."""
    pts = []
    h = spacing * np.sqrt(3.0) / 2.0
    for i, n in enumerate(rows):
        y = -dy + i * h
        # rows of odd length offset by half spacing
        base = (0.5 if (n % 2 == 1) else 0.0) * spacing
        xs = np.arange(n) * spacing + base
        # center this row horizontally
        xs = xs - (xs.max() + xs.min()) / 2.0 + xoff
        for x in xs:
            pts.append([x, y])
    return np.array(pts)


def _neg_score(params, rows):
    spacing, dy, xoff = params
    if spacing <= 0:
        return 1.0
    pts = _make_hex(rows, spacing, dy, xoff)
    return -_score(pts)


def construct():
    import time
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s

    best_pts = None
    best_score = -1.0

    # Candidate row-length patterns that sum to 50.
    # Alternating 7/6 patterns plus a short final row.
    patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4],   # 50
        [6, 7, 6, 7, 6, 7, 6, 5],   # 50
        [7, 6, 7, 6, 7, 6, 7, 4],
        [8, 6, 7, 6, 7, 6, 7, 3],   # 50
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 7, 6, 7, 6, 7, 6, 4],   # 50
        [5, 7, 6, 7, 6, 7, 6, 6],   # 50
    ]

    for rows in patterns:
        if sum(rows) != N:
            continue
        if time.time() > deadline - 5.0:
            break

        # Initial guess: spacing ~ 1/max_row_len
        max_n = max(rows)
        init = [1.0 / max_n, 0.0, 0.0]

        def obj(p, rows=rows):
            return _neg_score(p, rows)

        res = minimize(obj, init, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-9,
                                "adaptive": True})
        if time.time() > deadline - 5.0:
            break

        s = -res.fun
        if s > best_score:
            best_score = s
            best_pts = _make_hex(rows, *res.x)

        # Local polish with L-BFGS-B on a finer grid of starting offsets
        for xoff0 in (-0.02, 0.0, 0.02):
            for dy0 in (-0.01, 0.0, 0.01):
                if time.time() > deadline - 5.0:
                    break
                init2 = [res.x[0], dy0, xoff0]
                try:
                    res2 = minimize(obj, init2, method="L-BFGS-B",
                                    options={"maxiter": 500, "gtol": 1e-8})
                    s2 = -res2.fun
                    if s2 > best_score:
                        best_score = s2
                        best_pts = _make_hex(rows, *res2.x)
                except Exception:
                    pass

    if best_pts is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_pts
# EVOLVE-BLOCK-END
