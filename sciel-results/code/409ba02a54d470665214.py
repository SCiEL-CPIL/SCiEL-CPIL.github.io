"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull


_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - 3.0 - _elapsed()  # 3 s safety margin


def _normalize(P):
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    P = P / norms
    # ensure strictly nonzero
    P[np.abs(P) < 1e-12] = 1e-12
    return P


def _facet_circumradius(p1, p2, p3):
    """Circumradius of a triangle of unit vectors, in radians.
    For three points on the unit sphere, the circumradius of their spherical
    triangle equals the angular radius of the smallest spherical cap through
    them. For a facet of the convex hull of unit points, the circumradius of the
    planar triangle through the origin gives the angular covering radius of that
    empty cap: it is the angle from the origin to the circumscribed circle.
    For unit vectors the circumradius (planar) R satisfies R = a/(2 sin A) where
    a is a side and A the opposite angle; the angular radius is arcsin(R/|c|)...
    Simpler: the circumcenter of the planar triangle lies along the normal; the
    angular radius = angle between the origin and the circumscribed circle.
    For unit points, the planar circumradius R_c satisfies sin(theta) = R_c / (distance from origin to plane).
    We compute directly via the formula for the circumradius of the spherical triangle.
    """
    a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
    b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
    c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
    # spherical circumradius
    s = 0.5 * (a + b + c)
    # Use formula: tan(R) = (4 sin(s/2) sin((s-a)/2) sin((s-b)/2) sin((s-c)/2)) / (cos(a/2)cos(b/2)cos(c/2) - sin(a/2)sin(b/2)sin(c/2))
    num = 4.0 * np.sin(s / 2) * np.sin((s - a) / 2) * np.sin((s - b) / 2) * np.sin((s - c) / 2)
    den = (np.cos(a / 2) * np.cos(b / 2) * np.cos(c / 2)
           - np.sin(a / 2) * np.sin(b / 2) * np.sin(c / 2))
    if abs(den) < 1e-12:
        # degenerate / obtuse-ish; cap radius ~ max half-angle
        return 0.5 * max(a, b, c)
    R = np.arctan2(num, den)
    return R


def _covering_radius(P):
    """Max facet circumradius in radians (proxy for true covering radius)."""
    P = _normalize(P)
    try:
        hull = ConvexHull(P)
    except Exception:
        return np.pi
    worst = 0.0
    for simplex in hull.simplices:
        tri = P[simplex]
        R = _facet_circumradius(tri[0], tri[1], tri[2])
        if R > worst:
            worst = R
    return worst


def _cap_center(p1, p2, p3):
    """Point on the sphere at the center of the empty cap for a facet (circumcenter direction)."""
    # normal of the plane
    n = np.cross(p2 - p1, p3 - p1)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return (p1 + p2 + p3) / np.linalg.norm(p1 + p2 + p3)
    n = n / nn
    # ensure outward (away from origin side): the facet's outward normal points away from centroid of all points
    # For a convex hull of unit points, outward normal = n if n dot centroid > 0 else -n
    return n


def _grid_covering_radius(P, nlat=90, nlon=180):
    """Covering radius using a lat/lon grid (proxy, radians)."""
    P = _normalize(P)
    lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)
    lon = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
    best = 0.0
    for la in lat:
        cl = np.cos(la)
        sl = np.sin(la)
        for lo in lon:
            y = np.array([cl * np.cos(lo), cl * np.sin(lo), sl])
            d = np.arccos(np.clip(P @ y, -1, 1))
            m = d.min()
            if m > best:
                best = m
    return best


def _spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _icosahedron():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    pts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _optimize(P, max_iter=4000, step0=0.15, step_min=0.002, decay=0.998,
              repulse=0.05, repulse_r=0.9):
    """Iterative optimization: repulsion + worst-facet correction."""
    P = _normalize(P).copy()
    n = len(P)
    best = _covering_radius(P)
    bestP = P.copy()
    step = step0
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        # --- repulsion: push close points apart ---
        # pairwise angular distance
        G = P @ P.T
        G = np.clip(G, -1.0, 1.0)
        ang = np.arccos(G)
        # force: for pairs closer than repulse_r, push apart
        F = np.zeros_like(P)
        mask = (ang < repulse_r) & (np.arange(n)[:, None] != np.arange(n)[None, :])
        # strength grows as distance shrinks
        w = np.where(mask, (repulse_r - ang) ** 2, 0.0)
        # direction: move i away from j
        for i in range(n):
            js = np.where(mask[i])[0]
            if len(js) == 0:
                continue
            # sum of (P[i] - P[j]) weighted
            diff = P[i] - P[js]
            diff = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12)
            F[i] += (w[i, js] * diff).sum(axis=0)
        P = P + step * F
        P = _normalize(P)

        # --- worst-facet correction: move nearest point toward worst cap center ---
        try:
            hull = ConvexHull(P)
        except Exception:
            continue
        worstR = 0.0
        worstTri = None
        for simplex in hull.simplices:
            tri = P[simplex]
            R = _facet_circumradius(tri[0], tri[1], tri[2])
            if R > worstR:
                worstR = R
                worstTri = simplex
        if worstTri is None:
            continue
        center = _cap_center(P[worstTri[0]], P[worstTri[1]], P[worstTri[2]])
        # find the point nearest to the cap center (the one "responsible")
        d = np.arccos(np.clip(P @ center, -1, 1))
        j = int(np.argmin(d))
        # move it toward the center
        P[j] = P[j] + step * 1.5 * center
        P = _normalize(P)

        cur = _covering_radius(P)
        if cur < best - 1e-9:
            best = cur
            bestP = P.copy()
        step = max(step_min, step * decay)
    return bestP, best


def _local_polish(P, iters=3000, step0=0.05, step_min=0.001, decay=0.999):
    """Gradient-free polish minimizing grid covering radius."""
    P = _normalize(P).copy()
    best = _grid_covering_radius(P)
    bestP = P.copy()
    step = step0
    for it in range(iters):
        if _remaining() < 4.0:
            break
        # find worst grid point
        # (reuse grid)
        nlat, nlon = 60, 120
        lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        worst_y = None
        worst_d = -1.0
        worst_j = None
        for la in lat:
            cl = np.cos(la); sl = np.sin(la)
            for lo in lon:
                y = np.array([cl * np.cos(lo), cl * np.sin(lo), sl])
                d = np.arccos(np.clip(P @ y, -1, 1))
                j = int(np.argmin(d))
                if d[j] > worst_d:
                    worst_d = d[j]
                    worst_y = y
                    worst_j = j
        if worst_y is None:
            break
        # move the nearest point toward the worst location
        P[worst_j] = P[worst_j] + step * worst_y
        P = _normalize(P)
        cur = _grid_covering_radius(P)
        if cur < best - 1e-9:
            best = cur
            bestP = P.copy()
        step = max(step_min, step * decay)
    return bestP, best


def construct():
    global _START
    _START = time.time()
    bestP = _spiral(N)
    bestScore = _covering_radius(bestP)

    # candidate initializers
    inits = [
        _spiral(N, 0.5),
        _spiral(N, 0.0),
        _random_sphere(N, 1),
        _random_sphere(N, 7),
        _random_sphere(N, 42),
    ]
    # build an icosahedron-based 50-point set by adding scaled copies
    ico = _icosahedron()
    # create 50 by taking ico and adding small perturbations / extra symmetric points
    extra = []
    for k in range(4):
        ang = 2 * np.pi * k / 4
        rot = np.array([[np.cos(ang), -np.sin(ang), 0],
                        [np.sin(ang), np.cos(ang), 0],
                        [0, 0, 1.0]])
        extra.append(rot @ ico[0])
    # that's only 16; instead just use random for the rest
    inits.append(_random_sphere(N, 123))

    for init in inits:
        if _remaining() < 20.0:
            break
        init = _normalize(init)
        # quick score
        s0 = _covering_radius(init)
        if s0 < bestScore:
            bestScore = s0
            bestP = init.copy()
        # optimize
        P, sc = _optimize(init)
        if sc < bestScore:
            bestScore = sc
            bestP = P.copy()

    # final polish on best
    if _remaining() > 20.0:
        P, sc = _local_polish(bestP)
        if sc < bestScore:
            bestScore = sc
            bestP = P.copy()

    # one more polish pass if time
    if _remaining() > 15.0:
        P, sc = _local_polish(bestP, iters=2000, step0=0.03)
        if sc < bestScore:
            bestScore = sc
            bestP = P.copy()

    return _normalize(bestP)
# EVOLVE-BLOCK-END
