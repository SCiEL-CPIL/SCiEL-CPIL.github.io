"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

_DEADLINE = time.time() + 110.0  # safety margin vs 120 s kill


def _to_sphere(pts):
    """Normalize points to the unit sphere (robust to near-zero norms)."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _facet_circumradius(a, b, c):
    """Angular radius of the empty-cap circumcircle of spherical triangle (a,b,c).

    The circumcenter lies on the plane normal n = (b-a) x (c-a); it is +/- n
    (unit). We pick the sign whose distances to a,b,c are equal and smallest.
    """
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.pi
    n = n / nn
    best = np.pi
    for s in (1.0, -1.0):
        p = s * n
        da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
        db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
        dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
        if abs(da - db) < 1e-4 and abs(da - dc) < 1e-4:
            if da < best:
                best = da
    return best


def _covering_radius(pts):
    """Covering radius (radians): max facet circumradius of the convex hull.

    Returns pi for invalid configs (coincident points / degenerate hull).
    """
    pts = _to_sphere(pts)
    # coincident points
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    if np.min(d2[np.triu_indices(len(pts), k=1)]) < 1e-14:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    if len(hull.simplices) == 0:
        return np.pi
    # origin must be inside the hull (not contained in a closed hemisphere)
    # check via signed volumes of facets vs origin: all same sign => hemisphere
    s = hull.simplices
    maxr = 0.0
    for idx in s:
        r = _facet_circumradius(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if r > maxr:
            maxr = r
    return maxr


def _fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_pts():
    v = np.random.randn(N, 3)
    return _to_sphere(v)


def _refine(pts, step0=0.05):
    """Local greedy refinement: nudge each point toward its local farthest
    empty-cap circumcenter to reduce the covering radius."""
    pts = _to_sphere(pts.copy())
    best_cr = _covering_radius(pts)
    step = step0
    for it in range(60):
        if time.time() > _DEADLINE - 5.0:
            break
        improved = False
        # find the worst facet circumcenter and push nearest point toward it
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        worst_c = None
        worst_r = 0.0
        for idx in hull.simplices:
            a, b, c = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            for sgn in (1.0, -1.0):
                p = sgn * n
                da = np.arccos(np.clip(np.dot(p, a), -1, 1))
                db = np.arccos(np.clip(np.dot(p, b), -1, 1))
                dc = np.arccos(np.clip(np.dot(p, c), -1, 1))
                if abs(da - db) < 1e-4 and abs(da - dc) < 1e-4 and da > worst_r:
                    worst_r = da
                    worst_c = p
        if worst_c is None:
            break
        # nearest point to worst_c
        d = np.sum((pts - worst_c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        new = pts[k] + step * worst_c
        new = _to_sphere(new.reshape(1, 3))[0]
        cand = pts.copy()
        cand[k] = new
        cr = _covering_radius(cand)
        if cr < best_cr - 1e-9:
            pts = cand
            best_cr = cr
            improved = True
        else:
            step *= 0.8
        if not improved and step < 1e-4:
            break
    return pts, best_cr


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 110.0
    rng = np.random.default_rng(12345)

    # candidate starts
    starts = [_fibonacci()]
    for _ in range(6):
        starts.append(_random_pts())

    best_pts = None
    best_cr = np.inf
    for st in starts:
        if time.time() > _DEADLINE - 15.0:
            break
        p, cr = _refine(st, step0=0.06)
        if cr < best_cr:
            best_cr = cr
            best_pts = p

    if best_pts is None:
        return _fibonacci()
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
