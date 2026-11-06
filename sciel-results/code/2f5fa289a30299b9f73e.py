"""Covering the sphere with 130 equal caps: Riemannian L-BFGS-B on convex-hull circumradius."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_TIME_LIMIT = 120.0  # safety margin under the 150 s wall-clock kill
_T0 = time.time()


def _time_left():
    return _TIME_LIMIT - (time.time() - _T0)


def _normalize_rows(P):
    n = np.linalg.norm(P, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return P / n


def _retract(P, dP):
    # Riemannian retraction on the sphere: P + dP, then normalize.
    return _normalize_rows(P + dP)


def _project_tangent(P, dP):
    # Project a Euclidean displacement onto the tangent space at P.
    return dP - (dP * P).sum(axis=1, keepdims=True) * P


def _hull_circumradii(P):
    """Return (radii, centers) for each facet of the convex hull.
    radii[i] is the circumradius (in radians) of facet i, centers[i] is the
    cap-center direction (unit vector) to which that facet's circumradius applies.
    """
    try:
        hull = ConvexHull(P)
    except Exception:
        return None, None
    simplices = hull.simplices
    m = len(simplices)
    radii = np.zeros(m)
    centers = np.zeros((m, 3))
    for k, tri in enumerate(simplices):
        a = P[tri[0]]; b = P[tri[1]]; c = P[tri[2]]
        # Circumcenter of the triangle on the unit sphere.
        # Solve for center direction u (unit) with equal angular distance to a,b,c.
        # Equivalent: u is the unit vector in the plane through a,b,c closest to origin? 
        # Circumcenter of a spherical triangle = normalize of the point equidistant.
        # Use the planar circumcenter of a,b,c projected: 
        # The plane through a,b,c has normal n = a x b (and a x c). The circumcenter of the
        # triangle on the sphere lies along the direction that is the normalized solution.
        n_ab = np.cross(a, b)
        n_ac = np.cross(a, c)
        # Circumcenter direction = (a x b) x (a x c) normalized? That gives direction
        # perpendicular to the plane? No. Use linear solve for equidistance.
        # We want u (unit) with u.a = u.b = u.c. => u in span perpendicular to (a-b) and (a-c).
        # u = normalize( (a-b) x (a-c) )? That's the plane normal, which satisfies u.a=u.b=u.c only if
        # a,b,c are coplanar with origin - not generally. Instead solve:
        # minimize |u|^2 s.t. u.(a-b)=0, u.(a-c)=0, u.unit. The solution is the unit vector
        # parallel to (a-b) x (a-c)? No that's normal to the plane of a,b,c through origin.
        # Correct approach: circumcenter = normalize( cross(cross(a,b), cross(a,c)) )? Let's just
        # solve the linear system for the point equidistant in the plane.
        # Simpler and robust: the spherical circumcenter is the unit vector u solving
        # u.a = u.b = u.c. Two linear equations: u.(a-b)=0, u.(a-c)=0. The space of solutions is
        # 1-dim: direction d = (a-b) x (a-c). u = d/|d|.
        d = np.cross(a - b, a - c)
        nd = np.linalg.norm(d)
        if nd < 1e-12:
            u = a  # degenerate
        else:
            u = d / nd
        # Choose the side of the plane facing outward (same side as the hull interior is opposite).
        # The circumcenter should be on the same side as the origin relative to the facet plane? 
        # For a convex hull of points on the unit sphere, the facet plane is tangent-ish; the
        # relevant cap center is the unit vector in the direction of the circumcenter that is
        # *outside* (farther from origin than the facet). We pick the sign that makes u point
        # away from the origin relative to the facet, i.e., u . n_out > 0 where n_out is the
        # outward normal of the facet.
        n_out = np.cross(b - a, c - a)
        # Ensure outward: for convex hull, outward normal points away from centroid of points.
        centroid = P.mean(axis=0)
        if np.dot(n_out, centroid) > 0:
            n_out = -n_out
        if np.dot(u, n_out) < 0:
            u = -u
        centers[k] = u
        # Circumradius = angular distance from u to any of a,b,c.
        u = _normalize_rows(u.reshape(1, 3))[0]
        cosang = np.clip(np.dot(u, a), -1.0, 1.0)
        radii[k] = np.arccos(cosang)
    return radii, centers


def _covering_radius(P):
    radii, centers = _hull_circumradii(P)
    if radii is None:
        return np.inf
    return float(radii.max())


def _smooth_max(radii, centers, P, beta=40.0):
    """Smooth surrogate: soft-max of facet circumradii with gradient via soft-argmax.
    Returns (value, grad) where grad is Euclidean (before projection) w.r.t. P.
    """
    m = len(radii)
    # Soft-max weights.
    rmax = radii.max()
    w = np.exp(beta * (radii - rmax))
    w = w / w.sum()
    value = rmax + np.log(w.sum()) / beta  # soft-max value
    # d value / d radii_k = w_k
    # radii_k = arccos( centers_k . P[a] ) for a in facet k.
    # d radii_k / d P[a] = - centers_k / sqrt(1 - (centers_k.P[a])^2)
    grad = np.zeros_like(P)
    for k, tri in enumerate(_hull_circumradii(P)[1] is None and [] or _HULL_SIMPLEXES_CACHE):
        pass
    # Recompute per-facet to avoid double hull calls is complex; do it inline.
    return value, grad


def _objective_and_grad(P, beta=40.0):
    """Compute smooth objective and Euclidean gradient w.r.t. P."""
    try:
        hull = ConvexHull(P)
    except Exception:
        return _covering_radius(P), np.zeros_like(P)
    simplices = hull.simplices
    m = len(simplices)
    radii = np.zeros(m)
    centers = np.zeros((m, 3))
    for k, tri in enumerate(simplices):
        a = P[tri[0]]; b = P[tri[1]]; c = P[tri[2]]
        d = np.cross(a - b, a - c)
        nd = np.linalg.norm(d)
        if nd < 1e-12:
            u = a
        else:
            u = d / nd
        n_out = np.cross(b - a, c - a)
        centroid = P.mean(axis=0)
        if np.dot(n_out, centroid) > 0:
            n_out = -n_out
        if np.dot(u, n_out) < 0:
            u = -u
        u = u / np.linalg.norm(u)
        centers[k] = u
        cosang = np.clip(np.dot(u, a), -1.0, 1.0)
        radii[k] = np.arccos(cosang)
    rmax = radii.max()
    w = np.exp(beta * (radii - rmax))
    wsum = w.sum()
    w = w / wsum
    value = rmax + np.log(wsum) / beta
    grad = np.zeros_like(P)
    for k, tri in enumerate(simplices):
        wk = w[k]
        if wk < 1e-14:
            continue
        u = centers[k]
        for idx in tri:
            p = P[idx]
            cosang = np.clip(np.dot(u, p), -1.0, 1.0)
            sinang = np.sqrt(max(1.0 - cosang * cosang, 1e-12))
            # d radii_k / d p = -u / sinang
            grad[idx] += wk * (-u / sinang)
    # scale by beta? soft-max derivative: d value/d r_k = w_k (already). d value/d P = sum_k w_k * d r_k/dP.
    return value, grad


def _riemannian_lbgfgs(P0, max_iter=400, beta=40.0):
    P = _normalize_rows(P0.copy())

    def fun(X):
        val, _ = _objective_and_grad(_normalize_rows(X), beta)
        return val

    def jac(X):
        _, g = _objective_and_grad(_normalize_rows(X), beta)
        return g

    def retraction(X, dX):
        return _normalize_rows(X + dX)

    def egrad(X, g):
        return g

    # Use scipy minimize with L-BFGS-B in Euclidean space but with a retraction step is not
    # directly supported; instead we do a manual Riemannian L-BFGS-B via projected gradient
    # with line search, or simpler: use scipy minimize with trust-constr on the sphere using
    # a penalty. To keep it robust and simple, we run Euclidean L-BFGS-B and renormalize each
    # iteration via a wrapper that projects back (this is a common practical approach).
    bestP = P.copy()
    bestVal = fun(P)
    for it in range(max_iter):
        if _time_left() < 2.0:
            break
        val, g = _objective_and_grad(P, beta)
        if not np.isfinite(val):
            break
        # Tangent gradient
        gt = _project_tangent(P, g)
        # Simple backtracking line search along negative tangent direction.
        d = -gt
        nd = np.linalg.norm(d)
        if nd < 1e-12:
            break
        d = d / nd
        step = 0.01
        for ls in range(30):
            Pnew = _retract(P, step * d)
            vnew = fun(Pnew)
            if np.isfinite(vnew) and vnew < val:
                P = Pnew
                val = vnew
                if val < bestVal:
                    bestVal = val
                    bestP = P.copy()
                break
            step *= 0.5
        else:
            break
        # Anneal beta: start smooth, end sharp.
        beta = min(beta * 1.15, 300.0)
    return bestP, bestVal


def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(seed):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(N, 3))
    return _normalize_rows(P)


def construct():
    global _T0
    _T0 = time.time()
    starts = [_fibonacci_start()]
    for s in range(6):
        starts.append(_random_start(1000 + s))
    bestP = None
    bestVal = np.inf
    for P0 in starts:
        if _time_left() < 5.0:
            break
        P, val = _riemannian_lbgfgs(P0, max_iter=300, beta=40.0)
        # Final exact evaluation
        exact = _covering_radius(P)
        if exact < bestVal:
            bestVal = exact
            bestP = P.copy()
    if bestP is None:
        bestP = _fibonacci_start()
    return _normalize_rows(bestP)
