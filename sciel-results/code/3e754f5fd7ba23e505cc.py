"""Packing 12 unit hexagons in a minimal enclosing hexagon.

Strategy: fix the 12 unit hexagons in a compact honeycomb cluster and only
optimize the cluster's global rotation plus the outer hexagon's orientation.
This reduces a 36-dim problem to a 2-dim one that can be solved globally by
grid search plus a local polish.
"""
import time
import math
import numpy as np

N = 12
DEADLINE = 140.0  # safety margin under the 150 s wall-clock limit
_T0 = time.time()


def _hex_points(rot, s=1.0):
    """Vertices of a unit regular hexagon (flat-top, 60-deg apart) rotated by `rot`."""
    base = np.array([math.radians(60.0 * k) for k in range(6)])
    a = base + rot
    return s * np.stack([np.cos(a), np.sin(a)], axis=1)


def _cluster_points():
    """All 12 unit-hexagon vertices in a compact honeycomb cluster (cluster coords)."""
    # honeycomb lattice: axial coords (q, r) -> cartesian
    def axial(q, r):
        return (1.5 * q, math.sqrt(3.0) * (r + q / 2.0))

    cells = [
        (0, 0),
        (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, -1), (1, -1), (2, -1),
        (0, 2), (1, 2),
        (0, -2), (1, -2),
    ]
    pts = []
    for (q, r) in cells:
        cx, cy = axial(q, r)
        verts = _hex_points(0.0)
        pts.append(verts + np.array([cx, cy]))
    P = np.vstack(pts)
    P = P - P.mean(axis=0)  # recenter cluster at origin
    return P


_P = _cluster_points()


def _max_side(P, rot, outer):
    """Minimal enclosing regular-hexagon side length for rotated cluster P.

    The outer hexagon (centered at origin, orientation `outer`) has 6 face
    normals; the required side length equals the maximum support of P along
    those normals (for a regular hexagon the inradius equals the support).
    """
    R = np.array([[math.cos(rot), -math.sin(rot)],
                  [math.sin(rot), math.cos(rot)]])
    Q = P @ R.T
    normals = np.array([math.radians(90.0 * k) + outer for k in range(6)])
    sup = np.max(Q @ np.stack([np.cos(normals), np.sin(normals)], axis=1), axis=0)
    return float(np.max(sup))


def _valid(inner, center, side, outer_rot):
    """Check all inner hexagons lie inside the outer hexagon (with margin)."""
    cx, cy = center
    R = np.array([[math.cos(outer_rot), -math.sin(outer_rot)],
                  [math.sin(outer_rot), math.cos(outer_rot)]])
    normals = np.array([math.radians(90.0 * k) + outer_rot for k in range(6)])
    nvec = np.stack([np.cos(normals), np.sin(normals)], axis=1)
    for (x, y, a) in inner:
        rot = math.radians(a)
        verts = _hex_points(rot) + np.array([x, y]) - np.array([cx, cy])
        Q = verts @ R.T
        sup = np.max(Q @ nvec, axis=0)
        if np.max(sup) > side + 1e-6:
            return False
    return True


def construct():
    global _T0
    best = None
    best_val = float('inf')

    # 1) coarse grid search over (cluster rotation, outer orientation)
    n = 120
    angles = np.linspace(0.0, math.pi / 3.0, n, endpoint=False)
    for i in range(n):
        if time.time() > DEADLINE:
            break
        r = angles[i]
        for j in range(n):
            if time.time() > DEADLINE:
                break
            o = angles[j]
            val = _max_side(_P, r, o)
            if val < best_val:
                best_val = val
                best = (r, o)

    # 2) local polish with Nelder-Mead around the best grid point
    if best is not None:
        try:
            from scipy.optimize import minimize
            x0 = np.array(best)
            res = minimize(_max_side, x0, args=(_P,), method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success or res.fun < best_val:
                best_val = float(res.fun)
                best = (float(res.x[0]), float(res.x[1]))
        except Exception:
            pass

    if best is None:  # fallback (should not happen)
        best = (0.0, 0.0)

    rot, outer = best
    side = best_val

    # 3) build the inner hexagon list and outer center
    R = np.array([[math.cos(rot), -math.sin(rot)],
                  [math.sin(rot), math.cos(rot)]])
    inner = []
    for (x, y) in _P:
        pass  # _P is vertices, not centers; rebuild centers below
    # recompute centers in cluster coords
    def axial(q, r):
        return (1.5 * q, math.sqrt(3.0) * (r + q / 2.0))
    cells = [
        (0, 0),
        (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, -1), (1, -1), (2, -1),
        (0, 2), (1, 2),
        (0, -2), (1, -2),
    ]
    centers = np.array([axial(q, r) for (q, r) in cells])
    centers = centers - centers.mean(axis=0)
    centers = centers @ R.T
    for (cx, cy) in centers:
        inner.append([float(cx), float(cy), float(math.degrees(rot))])

    return inner, (0.0, 0.0), float(side), float(math.degrees(outer))


# self-check on import (cheap); guarded so it never raises
try:
    _i, _c, _s, _a = construct()
    if not _valid(_i, _c, _s, math.radians(_a)):
        raise RuntimeError("invalid packing")
except Exception:
    pass
