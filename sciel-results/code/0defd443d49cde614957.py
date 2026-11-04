"""Voronoi-cell inradius packing for 26 circles in the unit square."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _inradii(anchors):
    """
    For each anchor i, its circle radius is the distance from anchor i to
    the nearest boundary of its Voronoi cell.  The Voronoi cell boundary
    consists of:
      - the 4 walls of the unit square
      - for each other anchor j, the perpendicular bisector between i and j
    The inradius is the minimum distance from anchor i to all these boundaries.
    """
    n = anchors.shape[0]
    radii = np.empty(n)
    for i in range(n):
        ax, ay = anchors[i]
        # distance to walls
        d_wall = min(ax, 1.0 - ax, ay, 1.0 - ay)
        d = d_wall
        for j in range(n):
            if i == j:
                continue
            bx, by = anchors[j]
            dx = bx - ax
            dy = by - ay
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-15:
                # coincident anchors: inradius is 0
                d = 0.0
                break
            # distance from anchor i to the perpendicular bisector of i-j
            # bisector is the set of points equidistant from i and j
            # distance from i to bisector = dist / 2
            d_bis = dist * 0.5
            if d_bis < d:
                d = d_bis
        radii[i] = d
    return radii


def _objective(anchors_flat):
    """Minimize negative sum of inradii."""
    anchors = anchors_flat.reshape(-1, 2)
    radii = _inradii(anchors)
    return -np.sum(radii)


def _is_valid(anchors, radii, tol=1e-9):
    """
    Verify that circles with the computed inradii are truly non-overlapping
    and inside the square.  In a Voronoi inradius construction this should
    always hold, but we double-check for numerical safety.
    """
    n = anchors.shape[0]
    for i in range(n):
        x, y, r = anchors[i, 0], anchors[i, 1], radii[i]
        if r < -tol:
            return False
        if x - r < -tol or 1.0 - x - r < -tol or y - r < -tol or 1.0 - y - r < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = anchors[i, 0] - anchors[j, 0]
            dy = anchors[i, 1] - anchors[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _hex_init(n):
    """Hexagonal lattice initialization scaled to fit n points in unit square."""
    # approximate hex grid
    rows = int(np.ceil(np.sqrt(n * 2.0 / np.sqrt(3))))
    cols = int(np.ceil(n / rows))
    pts = []
    for row in range(rows):
        for col in range(cols):
            if len(pts) >= n:
                break
            x = (col + 0.5 * (row % 2)) / cols
            y = (row + 0.5) / rows
            pts.append([x, y])
    pts = pts[:n]
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])


def _grid_init(n):
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) < n:
                pts.append([(i + 0.5) / side, (j + 0.5) / side])
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])


def _random_init(n, rng):
    return rng.random((n, 2))


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150 s kill

    best_score = 0.0
    best_anchors = None

    inits = [
        _hex_init(N),
        _grid_init(N),
    ]

    rng = np.random.default_rng(12345)
    for _ in range(6):
        inits.append(_random_init(N, rng))

    for init_anchors in inits:
        if time.time() > deadline - 30.0:
            break

        init_flat = init_anchors.ravel()
        init_score = -_objective(init_flat)
        if init_score > best_score:
            best_score = init_score
            best_anchors = init_anchors.copy()

        # Differential evolution for global exploration
        if time.time() < deadline - 20.0:
            try:
                bounds = [(0.0, 1.0)] * (2 * N)
                de_result = differential_evolution(
                    _objective,
                    bounds=bounds,
                    x0=init_flat,
                    seed=int(rng.integers(0, 2**31)),
                    maxiter=30,
                    popsize=12,
                    tol=1e-8,
                    polish=False,
                    updating='deferred',
                )
                de_score = -de_result.fun
                if de_score > best_score:
                    best_score = de_score
                    best_anchors = de_result.x.reshape(-1, 2).copy()
            except Exception:
                pass

        # Nelder-Mead polish on the smooth inradius objective
        if time.time() < deadline - 10.0:
            try:
                nm_result = minimize(
                    _objective,
                    best_anchors.ravel(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                             'adaptive': True},
                )
                nm_score = -nm_result.fun
                if nm_score > best_score:
                    best_score = nm_score
                    best_anchors = nm_result.x.reshape(-1, 2).copy()
            except Exception:
                pass

        # L-BFGS-B with numerical gradient on the smooth objective
        if time.time() < deadline - 5.0:
            try:
                lb_result = minimize(
                    _objective,
                    best_anchors.ravel(),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10},
                )
                lb_score = -lb_result.fun
                if lb_score > best_score:
                    best_score = lb_score
                    best_anchors = lb_result.x.reshape(-1, 2).copy()
            except Exception:
                pass

    if best_anchors is None:
        best_anchors = _grid_init(N)

    # Final: recompute inradii and validate
    radii = _inradii(best_anchors)
    # Clip tiny negatives from numerical noise
    radii = np.maximum(radii, 0.0)

    if not _is_valid(best_anchors, radii):
        # Fallback: shrink radii slightly to ensure validity
        scale = 0.999
        radii = radii * scale
        if not _is_valid(best_anchors, radii):
            # Last resort: grid with tiny circles
            side = int(np.ceil(np.sqrt(N)))
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    result = np.zeros((N, 3))
    result[:, 0] = best_anchors[:, 0]
    result[:, 1] = best_anchors[:, 1]
    result[:, 2] = radii

    # Final safety clip
    for i in range(N):
        r = result[i, 2]
        x = result[i, 0]
        y = result[i, 1]
        r = min(r, x, 1.0 - x, y, 1.0 - y)
        r = max(r, 1e-6)
        result[i, 2] = r
        result[i, 0] = np.clip(x, r, 1.0 - r)
        result[i, 1] = np.clip(y, r, 1.0 - r)

    return result
# EVOLVE-BLOCK-END
