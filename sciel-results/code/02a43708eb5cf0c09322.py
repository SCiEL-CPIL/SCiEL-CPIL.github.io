import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
_DEADLINE = time.time() + 145.0  # hard wall-clock guard (evaluator kills at 150s)


def _hex_verts(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k)
        x = s * math.cos(t)
        y = s * math.sin(t)
        out.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return out


def _enclosing_side(pts, theta):
    """Smallest side of a theta-rotated regular hexagon (centered at origin)
    containing all points (apothem-based, exact for a centered hexagon)."""
    c = math.cos(theta)
    s = math.sin(theta)
    # outward normals of the 6 faces (pointing to each edge)
    dirs = np.array([
        [c, s],
        [c - 0.8660254037844386 * s, 0.5 * c + 0.8660254037844386 * s],
        [-0.5 * c + 0.8660254037844386 * s, -0.5 * s - 0.8660254037844386 * c],
        [-c, -s],
        [-c + 0.8660254037844386 * s, -0.5 * c + 0.8660254037844386 * s],
        [0.5 * c - 0.8660254037844386 * s, 0.5 * s + 0.8660254037844386 * c],
    ])
    maxh = -1e18
    for p in pts:
        proj = np.dot(p, dirs)
        maxh = max(maxh, float(np.max(proj)))
    return maxh / 0.8660254037844386  # side = apothem / cos(30deg)


def _objective(theta):
    """Minimize enclosing-hexagon side over the 12 hexagon centers (and angles)."""
    def f(x):
        if time.time() > _DEADLINE:
            return 1e12
        pts = []
        for i in range(N):
            cx, cy, ang = x[3 * i], x[3 * i + 1], x[3 * i + 2]
            pts.extend(_hex_verts(cx, cy, 1.0, ang))
        return _enclosing_side(pts, theta)
    return f


def _honeycomb_centers():
    dx, dy = 3.0, 3.0 * math.sqrt(3.0) / 2.0
    centers = []
    for k in range(-2, 3):
        for j in range(-2, 3):
            centers.append((dx * k + 1.5 * j, dy * j))
    return centers


def construct():
    t0 = time.time()
    centers = _honeycomb_centers()
    if len(centers) < N:
        centers = [(3.0 * k, 0.0) for k in range(N)]
    centers = centers[:N]

    # baseline: axis-aligned honeycomb, enclosing hexagon side
    base_theta = 0.0
    base_pts = []
    for (cx, cy) in centers:
        base_pts.extend(_hex_verts(cx, cy, 1.0, 0.0))
    best_side = _enclosing_side(base_pts, base_theta)
    best_x = np.array([v for c in centers for v in (c[0], c[1], 0.0)])
    best_center = (0.0, 0.0)
    best_theta = base_theta

    def commit(x, theta):
        nonlocal best_side, best_x, best_center, best_theta
        # recompute true (max-based) side for this candidate
        pts = []
        for i in range(N):
            pts.extend(_hex_verts(x[3 * i], x[3 * i + 1], 1.0, x[3 * i + 2]))
        side = _enclosing_side(pts, theta)
        if side < best_side - 1e-12:
            best_side = side
            best_x = x.copy()
            best_theta = theta
            # outer hexagon centered at centroid of all inner vertices
            cx = float(np.mean([p[0] for p in pts]))
            cy = float(np.mean([p[1] for p in pts]))
            best_center = (cx, cy)

    commit(best_x, base_theta)

    try:
        for theta in (0.0, math.radians(30.0)):
            if time.time() > _DEADLINE - 20.0:
                break
            f = _objective(theta)
            x0 = best_x.copy()
            res = minimize(f, x0, method='BFGS',
                           options={'maxiter': 300, 'gtol': 1e-8, 'disp': False})
            if time.time() > _DEADLINE:
                break
            if res.x is not None and np.all(np.isfinite(res.x)):
                commit(res.x, theta)
            # polish
            if time.time() > _DEADLINE - 10.0:
                break
            res2 = minimize(f, res.x, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                     'maxls': 50, 'disp': False})
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                commit(res2.x, theta)
    except Exception:
        pass

    # Finalize: shift inner centers so the enclosing hexagon is centered at origin.
    # (Score = side length, invariant to translation; keep a clean centered layout.)
    inner = []
    for i in range(N):
        inner.append([float(best_x[3 * i]), float(best_x[3 * i + 1]),
                      float(best_x[3 * i + 2])])
    return inner, (0.0, 0.0), float(best_side), float(best_theta)
