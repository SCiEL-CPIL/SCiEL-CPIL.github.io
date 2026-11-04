"""Packing 12 unit hexagons inside a regular hexagon: minimize outer side length."""
import time
import numpy as np

N = 12
_SQRT3 = np.sqrt(3.0)
_COS60 = 0.5
_SIN60 = np.sin(np.pi / 6.0)

# flat-top unit hexagon vertex directions (angles 0,60,...,300)
_VDIR = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)])
# six normal directions of the outer hexagon (0,30,...,150)
_NDIR = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 180, 30)])
_COSN = _NDIR[:, 0]
_SINN = _NDIR[:, 1]
_COS2N = 2.0 * _COSN
_SIN2N = 2.0 * _SINN


def _outer_side(pts, theta):
    """Minimal regular-hexagon side length enclosing points `pts` (Mx2)."""
    ct = np.cos(theta)
    st = np.sin(theta)
    # project points onto the 6 outer normal directions (rotated by theta)
    x = pts[:, 0] * ct + pts[:, 1] * st
    y = -pts[:, 0] * st + pts[:, 1] * ct
    # min/max over the 3 axis pairs
    lo = np.minimum.reduce([x, -x, y, -y, _COSN[0] * x + _SINN[0] * y,
                            _COSN[1] * x + _SINN[1] * y])
    hi = np.maximum.reduce([x, -x, y, -y, _COSN[0] * x + _SINN[0] * y,
                            _COSN[1] * x + _SINN[1] * y])
    w = hi - lo
    return float(w.max() / (2.0 * _SIN60))


def _outer_side_theta(pts, theta):
    return _outer_side(pts, theta)


def _obj(params):
    """Outer side length for a full parameter vector (3N values)."""
    P = params.reshape(N, 3)
    cx = P[:, 0]
    cy = P[:, 1]
    ca = np.cos(P[:, 2])
    sa = np.sin(P[:, 2])
    # rotated vertices: (cx, cy) + R(a) @ VDIR
    vx = cx[:, None] + ca[:, None] * _VDIR[0] - sa[:, None] * _VDIR[1]
    vy = cy[:, None] + sa[:, None] * _VDIR[0] + ca[:, None] * _VDIR[1]
    pts = np.column_stack((vx.ravel(), vy.ravel()))
    return _outer_side(pts, 0.0)


def _optimal_theta(pts):
    """Find rotation minimizing enclosing width (ternary search over 60 deg)."""
    lo, hi = 0.0, np.pi / 3.0
    for _ in range(40):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _outer_side_theta(pts, m1) < _outer_side_theta(pts, m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def _refine(params):
    """Polish a parameter vector with L-BFGS-B; return (params, side)."""
    from scipy.optimize import minimize
    P = params.reshape(N, 3).copy()
    res = minimize(_obj, P.ravel(), method='L-BFGS-B',
                   options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-10})
    best = res.x
    bestv = _obj(best)
    # recenter so centroid is at origin (invariance)
    bc = best.reshape(N, 3)
    bc[:, 0] -= bc[:, 0].mean()
    bc[:, 1] -= bc[:, 1].mean()
    best = bc.ravel()
    bestv = _obj(best)
    # compute the true outer side with optimal rotation
    P2 = best.reshape(N, 3)
    cx, cy, ca, sa = P2[:, 0], P2[:, 1], np.cos(P2[:, 2]), np.sin(P2[:, 2])
    vx = cx[:, None] + ca[:, None] * _VDIR[0] - sa[:, None] * _VDIR[1]
    vy = cy[:, None] + sa[:, None] * _VDIR[0] + ca[:, None] * _VDIR[1]
    pts = np.column_stack((vx.ravel(), vy.ravel()))
    side = _outer_side(pts, _optimal_theta(pts))
    return best, side


def _cluster_params():
    """Deterministic 1+6+5 hexagonal-cluster arrangement (good warm start)."""
    centers = [(0.0, 0.0)]
    # ring of 6 around center at distance sqrt(3)
    for k in range(6):
        a = np.radians(30 + 60 * k)
        centers.append((_SQRT3 * np.cos(a), _SQRT3 * np.sin(a)))
    # 5 more in the second ring (partial)
    for k in range(5):
        a = np.radians(90 + 60 * k)
        centers.append((2.0 * _SQRT3 * np.cos(a), 2.0 * _SQRT3 * np.sin(a)))
    P = np.array([(c[0], c[1], 0.0) for c in centers])
    return P.ravel()


def construct():
    deadline = time.time() + 140.0
    best_params = _cluster_params()
    best_side = _obj(best_params)
    # compute true side with optimal rotation for the initial cluster
    P2 = best_params.reshape(N, 3)
    cx, cy, ca, sa = P2[:, 0], P2[:, 1], np.cos(P2[:, 2]), np.sin(P2[:, 2])
    vx = cx[:, None] + ca[:, None] * _VDIR[0] - sa[:, None] * _VDIR[1]
    vy = cy[:, None] + sa[:, None] * _VDIR[0] + ca[:, None] * _VDIR[1]
    pts = np.column_stack((vx.ravel(), vy.ravel()))
    best_side = _outer_side(pts, _optimal_theta(pts))

    # polish the cluster start
    if time.time() < deadline:
        try:
            p, s = _refine(best_params)
            if s < best_side:
                best_params, best_side = p, s
        except Exception:
            pass

    # multi-start random warm starts
    rng = np.random.default_rng(12345)
    while time.time() < deadline - 20.0:
        P = rng.uniform(-6.0, 6.0, size=(N, 3))
        P[:, 2] *= (np.pi / 3.0)
        v = _obj(P)
        if v < best_side * 1.3:
            if time.time() >= deadline - 20.0:
                break
            try:
                p, s = _refine(P.ravel())
                if s < best_side:
                    best_params, best_side = p, s
            except Exception:
                pass

    # final high-precision polish of the best
    if time.time() < deadline - 5.0:
        try:
            p, s = _refine(best_params)
            if s < best_side:
                best_params, best_side = p, s
        except Exception:
            pass

    # final high-precision polish again if time remains
    if time.time() < deadline - 2.0:
        try:
            p, s = _refine(best_params)
            if s < best_side:
                best_params, best_side = p, s
        except Exception:
            pass

    # build the answer: recompute optimal rotation of the outer hexagon
    P2 = best_params.reshape(N, 3)
    cx, cy, ca, sa = P2[:, 0], P2[:, 1], np.cos(P2[:, 2]), np.sin(P2[:, 2])
    vx = cx[:, None] + ca[:, None] * _VDIR[0] - sa[:, None] * _VDIR[1]
    vy = cy[:, None] + sa[:, None] * _VDIR[0] + ca[:, None] * _VDIR[1]
    pts = np.column_stack((vx.ravel(), vy.ravel()))
    theta = _optimal_theta(pts)
    side = _outer_side(pts, theta)

    # outer hexagon center = centroid of the point cloud projected to be centered
    # recenter points so their "center" (midpoint of enclosing box) is origin
    cxs = pts[:, 0].mean()
    cys = pts[:, 1].mean()
    # shift all inner centers by -(cxs, cys)
    P2[:, 0] -= cxs
    P2[:, 1] -= cys

    inner = [[float(P2[i, 0]), float(P2[i, 1]), float(np.degrees(P2[i, 2]))] for i in range(N)]
    return inner, (0.0, 0.0), float(side), float(np.degrees(theta))
