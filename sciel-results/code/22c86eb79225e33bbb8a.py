"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fib_sample(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _hull_data(p):
    """Facets, outward circumcenters, and circumradii (radians) of the convex hull."""
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return None, None, None
    n_f = len(facets)
    if n_f == 0:
        return None, None, None
    a = p[facets[:, 0]]
    b = p[facets[:, 1]]
    c = p[facets[:, 2]]
    n = np.cross(a - b, b - c)
    nn = np.linalg.norm(n, axis=1)
    valid = nn > 1e-12
    cc = np.zeros((n_f, 3))
    rad = np.zeros(n_f)
    if np.any(valid):
        ccv = n[valid] / nn[valid, None]
        d = np.sum(ccv * a[valid], axis=1)
        ccv[d < 0] *= -1
        cc[valid] = ccv
        rad[valid] = np.arccos(np.clip(np.sum(ccv * a[valid], axis=1), -1.0, 1.0))
    return facets, cc, rad


def _covering_radius_deg(p):
    """Exact covering radius in degrees via hull facets; sampling fallback."""
    _, _, rad = _hull_data(p)
    if rad is None or not np.any(rad > 0):
        g = _fib_sample(20000)
        dots = g @ p.T
        return float(np.degrees(np.arccos(np.clip(np.min(dots, axis=1), -1.0, 1.0)).max()))
    return float(np.degrees(rad.max()))


def _soft_max(p, beta=100.0):
    """Smooth surrogate for the max circumradius; 0 if hull degenerate."""
    X = _normalize(p.reshape(N, 3))
    _, _, rad = _hull_data(X)
    if rad is None or not np.any(rad > 0):
        return 0.0
    rmax = rad.max()
    return rmax + np.log(np.sum(np.exp(beta * (rad - rmax)))) / beta


def _soft_max_grid(p, grid, beta=100.0):
    """
    Smooth surrogate based on a fixed dense spherical grid.
    Computes angular distance from each grid point to its nearest configuration point,
    then applies log-sum-exp soft-max.
    """
    X = _normalize(p.reshape(N, 3))
    # dots[i, j] = dot(grid[i], X[j])
    dots = grid @ X.T
    # min_dot[i] = max over j of dot(grid[i], X[j])  -> closest point on sphere
    min_dot = np.max(dots, axis=1)
    # angular distance = arccos(min_dot)
    # We want soft-max of angular distances.
    # Since arccos is decreasing, soft-max of angles = -soft-min of cos(angles)?
    # Actually, let's just compute angles and do log-sum-exp on them.
    # angles[i] = arccos(min_dot[i])
    angles = np.arccos(np.clip(min_dot, -1.0, 1.0))
    a_max = angles.max()
    if a_max < 1e-15:
        return 0.0
    return a_max + np.log(np.sum(np.exp(beta * (angles - a_max)))) / beta


def _repulsion(points, alpha=0.05, decay=0.995, max_iter=200, deadline=None):
    points = points.copy()
    mask = np.eye(N, dtype=bool)
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        diff = points[:, None, :] - points[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        inv = np.zeros_like(d2)
        np.divide(1.0, d2, out=inv, where=~mask)
        forces = np.sum(diff * inv[:, :, None], axis=1)
        forces = forces - np.sum(forces * points, axis=1)[:, None] * points
        points = _normalize(points + alpha * forces)
        alpha *= decay
    return points


def _lloyd(config):
    try:
        hull = ConvexHull(config)
        facets = hull.simplices
    except Exception:
        return config
    a = config[facets[:, 0]]
    b = config[facets[:, 1]]
    c = config[facets[:, 2]]
    n = np.cross(a - b, b - c)
    nn = np.linalg.norm(n, axis=1)
    valid = nn > 1e-12
    if not np.any(valid):
        return config
    cc = n[valid] / nn[valid, None]
    cc[np.sum(cc * a[valid], axis=1) < 0] *= -1
    nearest = np.argmax(cc @ config.T, axis=1)
    sums = np.zeros((N, 3))
    counts = np.zeros(N)
    np.add.at(sums, nearest, cc)
    np.add.at(counts, nearest, 1.0)
    new = config.copy()
    for i in range(N):
        if counts[i] > 0 and np.linalg.norm(sums[i]) > 1e-10:
            new[i] = sums[i] / np.linalg.norm(sums[i])
    return new


def _lbfqgs_hull(config, deadline):
    """L-BFGS on the soft-max surrogate of hull circumradii."""
    best = config.copy()
    best_score = _covering_radius_deg(config)
    for beta in (50.0, 80.0, 120.0, 200.0, 300.0):
        if time.time() > deadline - 8.0:
            break
        res = minimize(_soft_max, config.ravel(), args=(beta,), method='L-BFGS-B',
                       options={'maxiter': 400, 'maxls': 15, 'ftol': 1e-12,
                                'gtol': 1e-5, 'disp': False})
        cand = _normalize(res.x.reshape(N, 3))
        s = _covering_radius_deg(cand)
        if s < best_score - 1e-9:
            best, best_score = cand.copy(), s
            config = cand.copy()
    if time.time() < deadline - 8.0:
        res = minimize(_soft_max, config.ravel(), args=(30.0,), method='L-BFGS-B',
                       options={'maxiter': 300, 'maxls': 15, 'ftol': 1e-12,
                                'gtol': 1e-6, 'disp': False})
        cand = _normalize(res.x.reshape(N, 3))
        s = _covering_radius_deg(cand)
        if s < best_score - 1e-9:
            best, best_score = cand.copy(), s
    return best, best_score


def _lbfqgs_grid(config, grid, deadline):
    """L-BFGS on the soft-max surrogate of grid angular distances."""
    best = config.copy()
    best_score = _covering_radius_deg(config)
    for beta in (50.0, 100.0, 200.0, 400.0):
        if time.time() > deadline - 8.0:
            break
        res = minimize(_soft_max_grid, config.ravel(), args=(grid, beta), method='L-BFGS-B',
                       options={'maxiter': 400, 'maxls': 15, 'ftol': 1e-12,
                                'gtol': 1e-5, 'disp': False})
        cand = _normalize(res.x.reshape(N, 3))
        s = _covering_radius_deg(cand)
        if s < best_score - 1e-9:
            best, best_score = cand.copy(), s
            config = cand.copy()
    return best, best_score


def _greedy_polish(config, deadline):
    """Move the vertices of the worst facet toward its circumcenter (line search)."""
    config = config.copy()
    best_score = _covering_radius_deg(config)
    for _ in range(400):
        if time.time() > deadline - 3.0:
            break
        facets, cc, rad = _hull_data(config)
        if facets is None:
            break
        k = int(np.argmax(rad))
        target = cc[k]
        if np.linalg.norm(target) < 1e-8:
            break
        improved = False
        for t in (0.05, 0.02, 0.01, 0.005):
            trial = config.copy()
            for v in facets[k]:
                nv = config[v] + t * (target - config[v])
                trial[v] = nv / (np.linalg.norm(nv) + 1e-15)
            s = _covering_radius_deg(trial)
            if s < best_score - 1e-9:
                config, best_score, improved = trial, s, True
                break
        if not improved:
            break
    return config, best_score


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before the 150s kill

    pts = _normalize(_fib_sample(N))
    best = pts.copy()
    best_score = _covering_radius_deg(pts)

    # Phase 1: repulsion relaxation (fast, spreads points evenly)
    pts = _repulsion(pts, max_iter=200, deadline=deadline)
    s = _covering_radius_deg(pts)
    if s < best_score:
        best, best_score = pts.copy(), s
    if time.time() > deadline - 10.0:
        return _normalize(best)

    # Phase 2: Lloyd iterations (spherical-Voronoi balancing)
    for _ in range(80):
        if time.time() > deadline - 30.0:
            break
        new = _lloyd(pts)
        s = _covering_radius_deg(new)
        if s < best_score:
            best, best_score = new.copy(), s
        pts = new
    if time.time() > deadline - 10.0:
        return _normalize(best)

    # Phase 3a: L-BFGS on the hull-based soft-max surrogate
    pts, s = _lbfqgs_hull(pts, deadline)
    if s < best_score:
        best, best_score = pts.copy(), s
    if time.time() > deadline - 15.0:
        return _normalize(best)

    # Phase 3b: L-BFGS on the grid-based soft-max surrogate (differentiable, robust)
    # Use a moderately dense grid for the surrogate to keep it smooth but accurate
    grid = _fib_sample(10000)
    pts, s = _lbfqgs_grid(pts, grid, deadline)
    if s < best_score:
        best, best_score = pts.copy(), s
    if time.time() > deadline - 6.0:
        return _normalize(best)

    # Phase 4: greedy facet polish (exact objective)
    pts, s = _greedy_polish(pts, deadline)
    if s < best_score:
        best, best_score = pts.copy(), s

    return _normalize(best)
# EVOLVE-BLOCK-END
