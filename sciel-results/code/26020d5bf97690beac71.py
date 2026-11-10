"""Pack 300 equal circles in the unit disk: maximize the common radius.

Design: multi-start force-based relaxation (repulsion + confinement) with
annealing, followed by a high-precision coordinate-descent polish. An explicit
deadline guard returns the best valid configuration found within the budget.
"""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 300
DEADLINE = 150.0
MARGIN = 10.0
_START = time.time()

# Tunable constants
K = 200.0
ALPHA = 1.0
BETA = 1.0
LAMBDA = 10.0
GAMMA = 1.0
STEP = 0.05
ETA = 0.9
MIN_T = 0.05


def _time_left():
    return DEADLINE - (time.time() - _START) - MARGIN


def _score(pts):
    """Return the common radius r for a valid configuration, else -inf."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(pts)):
        return -np.inf
    # container containment
    rad = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(rad > 1.0):
        return -np.inf
    # pairwise separation
    tree = cKDTree(pts)
    pairs = tree.query_pairs(0.5)  # only pairs closer than 0.5 can violate
    if len(pairs) == 0:
        return min(1.0 - rad.max(), 0.5)
    dmin = np.inf
    for i, j in pairs:
        d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
        if d < dmin:
            dmin = d
    return min(dmin / 2.0, 1.0 - rad.max())


def _force(pts):
    """Total force per point: repulsion (pairwise) + confinement (radial)."""
    n = pts.shape[0]
    f = np.zeros((n, 2))
    tree = cKDTree(pts)
    pairs = tree.query_pairs(0.5)
    for i, j in pairs:
        dx = pts[i, 0] - pts[j, 0]
        dy = pts[i, 1] - pts[j, 1]
        d2 = dx * dx + dy * dy
        d = np.sqrt(d2)
        if d < 1e-12:
            # tiny random nudge to break exact coincidence
            dx = np.random.uniform(-1e-3, 1e-3)
            dy = np.random.uniform(-1e-3, 1e-3)
            d = np.hypot(dx, dy)
        w = K / (d2 + 1e-9)
        f[i, 0] += w * dx
        f[i, 1] += w * dy
        f[j, 0] -= w * dx
        f[j, 1] -= w * dy
    # confinement: pull toward the boundary when close to it
    rad = np.hypot(pts[:, 0], pts[:, 1])
    close = rad > 0.9
    if np.any(close):
        r = rad[close]
        f[close, 0] += BETA * (1.0 - r) * (pts[close, 0] / r)
        f[close, 1] += BETA * (1.0 - r) * (pts[close, 1] / r)
    return f


def _relax(pts, t0, t1, steps):
    """Annealed force relaxation. Returns the final point array."""
    pts = pts.copy()
    for s in range(steps):
        if _time_left() <= 0:
            return pts
        frac = s / max(1, steps - 1)
        t = t0 + (t1 - t0) * frac
        f = _force(pts)
        # annealed noise
        if t > MIN_T:
            f = f + GAMMA * t * np.random.randn(*pts.shape)
        step = min(STEP, t)
        pts = pts + step * f
        # enforce containment (project back inside)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        over = rad > 1.0
        if np.any(over):
            pts[over, 0] = pts[over, 0] / rad[over]
            pts[over, 1] = pts[over, 1] / rad[over]
    return pts


def _polish(pts):
    """Coordinate-descent local polish: move each point along the direction that
    most increases the minimum clearance, with a line search. Many passes."""
    pts = pts.copy()
    n = pts.shape[0]
    for _ in range(60):
        if _time_left() <= 0:
            break
        improved = False
        # iterate points in random order to reduce bias
        order = np.random.permutation(n)
        for i in order:
            # current min clearance involving point i
            tree = cKDTree(pts)
            dist, idx = tree.query(pts[i], k=1 + 1)
            # nearest neighbour distance
            dmin = dist[1] if len(idx) > 1 else np.inf
            r_cont = 1.0 - np.hypot(pts[i, 0], pts[i, 1])
            # objective to maximize: min(dmin/2, r_cont)
            # candidate directions: away from nearest neighbour, and toward center
            cand = []
            if dmin < np.inf:
                j = idx[1]
                d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                if d > 1e-12:
                    cand.append((pts[i, 0] - pts[j, 0]) / d, (pts[i, 1] - pts[j, 1]) / d)
            r = np.hypot(pts[i, 0], pts[i, 1])
            if r > 1e-12:
                cand.append((-pts[i, 0] / r, -pts[i, 1] / r))
            if not cand:
                continue
            best_val = min(dmin / 2.0, r_cont)
            best_step = None
            best_dir = None
            for dx, dy in cand:
                for step in (0.02, 0.01, 0.005, 0.002):
                    ni = pts[i].copy()
                    ni[0] += step * dx
                    ni[1] += step * dy
                    # check containment
                    nr = np.hypot(ni[0], ni[1])
                    if nr > 1.0:
                        continue
                    pts[i] = ni
                    d, ix = tree.query(ni, k=2)
                    ndmin = d[1] if len(ix) > 1 else np.inf
                    nrc = 1.0 - nr
                    val = min(ndmin / 2.0, nrc)
                    if val > best_val + 1e-12:
                        best_val = val
                        best_step = step
                        best_dir = (dx, dy)
            if best_dir is not None:
                pts[i, 0] += best_step * best_dir[0]
                pts[i, 1] += best_step * best_dir[1]
                improved = True
        if not improved:
            break
    return pts


def _initial_grid():
    """Hexagonal lattice inside the disk, then trim to N points."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) < N:
        # fallback: dense random
        pts = np.random.uniform(-1, 1, (N, 2))
        rad = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts * (0.99 / rad)[:, None]
        return pts
    # take N nearest to center
    order = np.argsort(radius, kind="stable")[:N]
    return points[order]


def construct():
    best = None
    best_score = -np.inf
    rng = np.random.RandomState(0)

    # First attempt from the grid (deterministic baseline)
    pts = _initial_grid()
    pts = _relax(pts, 0.5, 0.1, 400)
    pts = _polish(pts)
    s = _score(pts)
    if s > best_score:
        best_score = s
        best = pts.copy()

    # Multi-start: random initializations, each relaxed + polished
    while _time_left() > 5:
        # random start: uniform in disk, jittered
        pts = np.random.uniform(-1, 1, (N, 2))
        rad = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts * (0.99 / rad)[:, None]
        pts = _relax(pts, 0.6, 0.1, 300)
        pts = _polish(pts)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best = pts.copy()
            # extra polish on the improved configuration
            if _time_left() > 5:
                pts = _polish(best)
                s2 = _score(pts)
                if s2 > best_score:
                    best_score = s2
                    best = pts.copy()

    # Final safety: ensure validity
    if best is None:
        best = _initial_grid()
    s = _score(best)
    if not np.isfinite(s) or s <= 0:
        # last-resort valid fallback
        best = _initial_grid()
    return np.asarray(best, dtype=float)
