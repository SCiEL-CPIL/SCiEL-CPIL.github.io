"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _facet_circumcenter(p0, p1, p2):
    """Circumcenter (unit vector) of a spherical triangle on the unit sphere."""
    v1 = p1 - p0
    v2 = p2 - p0
    A = np.array([[v1 @ v1, v1 @ v2],
                  [v2 @ v1, v2 @ v2]])
    b = np.array([-(p0 @ v1), -(p0 @ v2)])
    try:
        ab = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    c = p0 + ab[0] * v1 + ab[1] * v2
    cnorm = np.linalg.norm(c)
    if cnorm < 1e-10:
        return None
    return c / cnorm


def _facet_angle(p0, p1, p2, c):
    """Angular radius of the facet = max angle from circumcenter to vertices."""
    a0 = np.arccos(np.clip(np.dot(c, p0), -1.0, 1.0))
    a1 = np.arccos(np.clip(np.dot(c, p1), -1.0, 1.0))
    a2 = np.arccos(np.clip(np.dot(c, p2), -1.0, 1.0))
    return max(a0, a1, a2)


def _covering_radius(pts):
    """Covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    max_ang = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        p0, p1, p2 = pts[simplex]
        c = _facet_circumcenter(p0, p1, p2)
        if c is None:
            continue
        max_ang = max(max_ang, _facet_angle(p0, p1, p2, c))
    return np.degrees(max_ang)


def _facet_data(pts):
    """Return list of (idx0, idx1, idx2, circumcenter, angle) for each hull facet."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    out = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        i0, i1, i2 = simplex
        p0, p1, p2 = pts[i0], pts[i1], pts[i2]
        c = _facet_circumcenter(p0, p1, p2)
        if c is None:
            continue
        ang = _facet_angle(p0, p1, p2, c)
        out.append((i0, i1, i2, c, ang))
    return out


def _smooth_max_value_and_grad(pts, beta=60.0):
    """
    Smooth-max (log-sum-exp) over facet angular radii, with an ANALYTIC
    gradient (chain rule through the circumcenter), on the sphere tangent space.
    Returns (value_in_degrees, grad (N,3) projected off radial direction).
    """
    data = _facet_data(pts)
    if not data:
        return 180.0, np.zeros_like(pts)
    angles = np.array([d[4] for d in data])
    m = angles.max()
    if m < 1e-12:
        return 0.0, np.zeros_like(pts)
    w = np.exp(angles - m)
    wsum = w.sum()
    pw = w / wsum  # softmax weights

    # analytic gradient of each facet angle w.r.t. its 3 vertices
    facet_grads = []
    for (i0, i1, i2, c, ang) in data:
        p0, p1, p2 = pts[i0], pts[i1], pts[i2]
        # angle = arccos(c . p_k); d/dp = -c / sqrt(1-(c.p)^2)
        g = np.zeros((3, 3))
        for k, p in enumerate((p0, p1, p2)):
            cp = np.clip(np.dot(c, p), -1.0 + 1e-12, 1.0 - 1e-12)
            denom = np.sqrt(max(1.0 - cp * cp, 1e-12))
            g[k] = -c / denom
        facet_grads.append((i0, i1, i2, g))

    grad = np.zeros_like(pts)
    for (i0, i1, i2, g), wi in zip(facet_grads, pw):
        grad[i0] += wi * g[0]
        grad[i1] += wi * g[1]
        grad[i2] += wi * g[2]

    # project to tangent space (remove radial component) for sphere constraint
    radial = (grad * pts).sum(axis=1, keepdims=True)
    grad = grad - radial * pts

    value = np.degrees(m + np.log(wsum) / beta)
    grad = grad / beta
    return value, grad


def _lbfgs_step(pts, beta, maxiter, deadline):
    from scipy.optimize import minimize
    state = {'beta': beta}

    def fun(x):
        p = _normalize(x.reshape(N, 3))
        return _smooth_max_value_and_grad(p, state['beta'])[0]

    def jac(x):
        p = _normalize(x.reshape(N, 3))
        return _smooth_max_value_and_grad(p, state['beta'])[1].flatten()

    try:
        res = minimize(
            fun, pts.flatten(), jac=jac, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10,
                     'maxfun': maxiter * 3}
        )
    except Exception:
        return pts
    p = _normalize(res.x.reshape(N, 3))
    return p


def _facet_repair(pts, n_steps=40, deadline=None):
    """
    Geometric repair: repeatedly move the 3 vertices of the worst facet toward
    their circumcenter (equalizing the facet's angular radius), then re-hull.
    """
    pts = _normalize(pts)
    for _ in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        data = _facet_data(pts)
        if not data:
            break
        # pick worst facet
        worst = max(data, key=lambda d: d[4])
        i0, i1, i2, c, ang = worst
        if ang < 1e-6:
            break
        p0, p1, p2 = pts[i0].copy(), pts[i1].copy(), pts[i2].copy()
        # move each vertex toward circumcenter by a small step
        step = 0.15
        for p in (p0, p1, p2):
            p = p + step * (c - p)
            pn = np.linalg.norm(p)
            if pn < 1e-12:
                continue
            p = p / pn
        pts[i0] = p0
        pts[i1] = p1
        pts[i2] = p2
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    # Fibonacci spiral initial points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_pts = _normalize(pts)
    best_radius = _covering_radius(best_pts)

    np.random.seed(42)
    n_restarts = 6

    for restart in range(n_restarts):
        if time.time() > deadline - 25.0:
            break

        if restart == 0:
            current = best_pts.copy()
        else:
            if restart < 4:
                current = best_pts + 0.25 * np.random.randn(N, 3)
            else:
                i2 = np.arange(N) + 0.5 + np.random.rand()
                z2 = 1.0 - 2.0 * np.minimum(i2, N) / N
                phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2
                r2 = np.sqrt(1.0 - z2 * z2)
                current = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
            current = _normalize(current)

        # Phase 1: analytic-grad L-BFGS on smooth-max (coarse)
        if time.time() < deadline - 20.0:
            current = _lbfgs_step(current, beta=60.0, maxiter=80, deadline=deadline - 15.0)

        # Phase 2: geometric facet repair
        if time.time() < deadline - 15.0:
            current = _facet_repair(current, n_steps=60, deadline=deadline - 10.0)

        # Phase 3: analytic-grad L-BFGS polish (fine)
        if time.time() < deadline - 10.0:
            current = _lbfgs_step(current, beta=120.0, maxiter=120, deadline=deadline - 5.0)

        # Phase 4: final facet repair
        if time.time() < deadline - 5.0:
            current = _facet_repair(current, n_steps=40, deadline=deadline - 3.0)

        cr = _covering_radius(current)
        if cr < best_radius:
            best_radius = cr
            best_pts = _normalize(current).copy()

    return _normalize(best_pts)


# EVOLVE-BLOCK-START
# (The code above is the implementation; this block marker is for the evolution system)
# EVOLVE-BLOCK-END
