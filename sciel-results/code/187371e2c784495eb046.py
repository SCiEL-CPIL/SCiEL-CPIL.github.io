"""Geometric covering-radius optimizer for 50 points on the unit sphere."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50
T_LIMIT = 120.0          # hard wall-clock budget (evaluator kills at 150s)
SAFETY = 10.0            # return best with this many seconds to spare


def _normalize(P):
    n = np.linalg.norm(P, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return P / n


def _covering_radius(P):
    """Covering radius in radians = max facet circumradius (empty-cap check)."""
    P = _normalize(P)
    try:
        hull = ConvexHull(P)
    except Exception:
        return np.inf
    best = 0.0
    for simplex in hull.simplices:
        pts = P[simplex]
        # circumcenter of the spherical triangle via planar circumcenter of the 3 points
        a, b, c = pts
        # use the plane through a,b,c; compute circumcenter in that plane
        ab = b - a
        ac = c - a
        cross = np.cross(ab, ac)
        nrm = np.linalg.norm(cross)
        if nrm < 1e-12:
            continue
        n = cross / nrm
        # project a,b,c to 2D in plane basis
        u = ab / np.linalg.norm(ab)
        v = np.cross(n, u)
        A = np.array([[u @ a, v @ a, 1.0],
                      [u @ b, v @ b, 1.0],
                      [u @ c, v @ c, 1.0]])
        B = np.array([np.linalg.norm(a) ** 2,
                      np.linalg.norm(b) ** 2,
                      np.linalg.norm(c) ** 2])
        try:
            sol = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            continue
        # circumcenter in 3D: O = a + sol[0]*(b-a) ... actually solve gives (x,y) in basis
        # Reconstruct: center = x*u + y*v + z*n where x,y are in-plane coords, z=0
        # Our A used (u@p, v@p) so sol[0],sol[1] are in-plane coords, sol[2] unused
        cx, cy = sol[0], sol[1]
        center = cx * u + cy * v
        # distance from center to a (planar) = circumradius R_planar
        R_planar = np.linalg.norm(center - a)
        # angular radius: the cap half-angle for this facet
        # circumradius on sphere: sin(theta) = R_planar / (something). 
        # For a spherical triangle with planar circumcenter at distance d from origin,
        # the angular radius = arccos( (d^2 - R_planar^2) / d^2 )? Simpler: 
        # The cap is centered at center/|center| with half-angle alpha where
        # cos(alpha) = (|center|^2 - R_planar^2) / |center|^2 ... 
        # Actually for a plane at distance h from origin, a circle of planar radius R,
        # the spherical cap half-angle alpha satisfies: sin^2(alpha) = R^2/(h^2+R^2)? 
        # Let's use: the angular distance from cap center to vertex = arccos( n_hat . a )
        h = np.linalg.norm(center)
        if h < 1e-12:
            continue
        n_hat = center / h
        # angular distance to vertex a
        cos_a = np.clip(n_hat @ a, -1.0, 1.0)
        ang = np.arccos(cos_a)
        if ang > best:
            best = ang
    return best


def _covering_radius_deg(P):
    return np.degrees(_covering_radius(P))


def _softmax_obj(P, beta):
    """Smooth surrogate: soft-max of pairwise nearest angular distances over a grid."""
    P = _normalize(P)
    # sample points on sphere (Fibonacci + random)
    m = 2000
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # angular distance matrix: cos of angle = Y @ P^T
    cosang = Y @ P.T
    cosang = np.clip(cosang, -1.0, 1.0)
    dist = np.arccos(cosang)          # (m, N)
    nearest = np.min(dist, axis=1)    # (m,)
    # soft-max
    beta = max(beta, 1e-3)
    shifted = beta * (nearest - nearest.max())
    smax = nearest.max() + np.log(np.sum(np.exp(shifted)) / m) / beta
    return smax


def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _lgbfgs_phase(P, deadline):
    """Smooth soft-max minimization via L-BFGS on Cartesian coords with re-normalization."""
    from scipy.optimize import minimize
    x0 = P.flatten()
    best = P.copy()
    best_val = _covering_radius_deg(best)

    def fun(x):
        Q = _normalize(x.reshape(N, 3))
        return _softmax_obj(Q, beta=20.0)

    try:
        res = minimize(fun, x0, method='L-BFGS-B',
                       options={'maxiter': 200, 'gtol': 1e-8, 'maxfun': 2000})
        Q = _normalize(res.x.reshape(N, 3))
        v = _covering_radius_deg(Q)
        if np.isfinite(v) and v < best_val:
            best, best_val = Q, v
    except Exception:
        pass
    return best, best_val, time.time() < deadline


def _facet_refine(P, deadline, iters=400):
    """Geometric refinement: move points toward circumcenters of the largest facets."""
    P = _normalize(P).copy()
    best = P.copy()
    best_val = _covering_radius_deg(best)
    step = 0.05
    for it in range(iters):
        if time.time() > deadline:
            break
        try:
            hull = ConvexHull(P)
        except Exception:
            break
        # find the facet with the largest circumradius (angular)
        worst_idx = -1
        worst_ang = 0.0
        worst_center = None
        for si, simplex in enumerate(hull.simplices):
            a, b, c = P[simplex]
            ab = b - a
            ac = c - a
            cross = np.cross(ab, ac)
            nrm = np.linalg.norm(cross)
            if nrm < 1e-12:
                continue
            n = cross / nrm
            u = ab / np.linalg.norm(ab)
            v = np.cross(n, u)
            A = np.array([[u @ a, v @ a, 1.0],
                          [u @ b, v @ b, 1.0],
                          [u @ c, v @ c, 1.0]])
            B = np.array([np.dot(a, a), np.dot(b, b), np.dot(c, c)])
            try:
                sol = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                continue
            center = sol[0] * u + sol[1] * v
            h = np.linalg.norm(center)
            if h < 1e-12:
                continue
            n_hat = center / h
            cos_a = np.clip(n_hat @ a, -1.0, 1.0)
            ang = np.arccos(cos_a)
            if ang > worst_ang:
                worst_ang = ang
                worst_idx = si
                worst_center = n_hat
        if worst_idx < 0:
            break
        # move the 3 vertices of the worst facet toward the cap center (geodesic)
        simplex = hull.simplices[worst_idx]
        for idx in simplex:
            p = P[idx]
            # geodesic step toward worst_center
            ang = np.arccos(np.clip(p @ worst_center, -1.0, 1.0))
            if ang < 1e-9:
                continue
            t = min(step, 0.5 * ang)
            newp = (np.sin(ang - t) / np.sin(ang)) * p + (np.sin(t) / np.sin(ang)) * worst_center
            P[idx] = newp
        P = _normalize(P)
        v = _covering_radius_deg(P)
        if np.isfinite(v) and v < best_val:
            best, best_val = P.copy(), v
    return best, best_val


def _polish(P, deadline):
    """High-precision final polish: small-step facet refinement + local L-BFGS."""
    P = _normalize(P).copy()
    best = P.copy()
    best_val = _covering_radius_deg(best)
    # tiny-step geometric refinement
    for _ in range(200):
        if time.time() > deadline:
            break
        try:
            hull = ConvexHull(P)
        except Exception:
            break
        worst_ang = 0.0
        worst_center = None
        worst_simplex = None
        for simplex in hull.simplices:
            a, b, c = P[simplex]
            ab = b - a
            ac = c - a
            cross = np.cross(ab, ac)
            nrm = np.linalg.norm(cross)
            if nrm < 1e-12:
                continue
            n = cross / nrm
            u = ab / np.linalg.norm(ab)
            v = np.cross(n, u)
            A = np.array([[u @ a, v @ a, 1.0],
                          [u @ b, v @ b, 1.0],
                          [u @ c, v @ c, 1.0]])
            B = np.array([np.dot(a, a), np.dot(b, b), np.dot(c, c)])
            try:
                sol = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                continue
            center = sol[0] * u + sol[1] * v
            h = np.linalg.norm(center)
            if h < 1e-12:
                continue
            n_hat = center / h
            ang = np.arccos(np.clip(n_hat @ a, -1.0, 1.0))
            if ang > worst_ang:
                worst_ang = ang
                worst_center = n_hat
                worst_simplex = simplex
        if worst_simplex is None:
            break
        for idx in worst_simplex:
            p = P[idx]
            ang = np.arccos(np.clip(p @ worst_center, -1.0, 1.0))
            if ang < 1e-9:
                continue
            t = min(0.01, 0.5 * ang)
            newp = (np.sin(ang - t) / np.sin(ang)) * p + (np.sin(t) / np.sin(ang)) * worst_center
            P[idx] = newp
        P = _normalize(P)
        v = _covering_radius_deg(P)
        if np.isfinite(v) and v < best_val:
            best, best_val = P.copy(), v
    # local L-BFGS polish on soft-max
    from scipy.optimize import minimize
    def fun(x):
        Q = _normalize(x.reshape(N, 3))
        return _softmax_obj(Q, beta=50.0)
    try:
        res = minimize(fun, best.flatten(), method='L-BFGS-B',
                       options={'maxiter': 100, 'gtol': 1e-10, 'maxfun': 500})
        Q = _normalize(res.x.reshape(N, 3))
        v = _covering_radius_deg(Q)
        if np.isfinite(v) and v < best_val:
            best, best_val = Q, v
    except Exception:
        pass
    return best, best_val


def construct():
    t0 = time.time()
    deadline = t0 + T_LIMIT - SAFETY

    # Phase 0: good start
    P = _fibonacci_start()
    best = P.copy()
    best_val = _covering_radius_deg(best)

    # Phase 1: L-BFGS on smooth surrogate
    if time.time() < deadline:
        P, v, ok = _lgbfgs_phase(P, deadline)
        if np.isfinite(v) and v < best_val:
            best, best_val = P, v

    # Phase 2: geometric facet refinement
    if time.time() < deadline:
        P, v = _facet_refine(P, deadline, iters=300)
        if np.isfinite(v) and v < best_val:
            best, best_val = P, v

    # Phase 3: final polish
    if time.time() < deadline:
        P, v = _polish(P, deadline)
        if np.isfinite(v) and v < best_val:
            best, best_val = P, v

    # safety: ensure finite, normalized, nonzero
    best = _normalize(best)
    if not np.all(np.isfinite(best)):
        best = _fibonacci_start()
    return best
