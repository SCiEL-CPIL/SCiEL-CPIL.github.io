# EVOLVE-BLOCK-START
import time
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:  # pragma: no cover - scipy is stated available
    _scipy_minimize = None

_T0 = time.time()
_DEADLINE = _T0 + 600.0
_SAFETY = 30.0          # seconds to stop before the wall-clock kill
_EPS = 1e-9
_SEP_TOL = 1e-6
_CONTAIN_TOL = 1e-6


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    t = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + d * np.cos(t), cy + d * np.sin(t)], axis=1)


def _inner_edges(v):
    return v - np.roll(v, 1, axis=0)


def _separation_gap(v1, v2):
    """Return min over both hexagons' edge axes of (gap between projection intervals).
    Positive => separated (disjoint); <=0 => touching/overlapping."""
    e1 = _inner_edges(v1)
    n1 = np.stack([-e1[:, 1], e1[:, 0]], axis=1)
    p1 = v1 @ n1.T
    mn1, mx1 = p1.min(0), p1.max(0)
    e2 = _inner_edges(v2)
    n2 = np.stack([-e2[:, 1], e2[:, 0]], axis=1)
    p2 = v2 @ n2.T
    mn2, mx2 = p2.min(0), p2.max(0)
    gap12 = mx1 - mn2
    gap21 = mx2 - mn1
    return float(min(gap12.min(), gap21.min()))


def _max_vertex_proj(v, c, n):
    """max_i n_i · (v_i - c) for the 6 edge normals n (rows)."""
    return float(np.max((v - c) @ n.T))


def _build_initial():
    r3 = np.sqrt(3.0) / 2.0
    x1 = 2.5
    y1 = 3.0 * r3
    x2 = 5.0
    y2 = 6.0 * r3
    # 12 unit hexagons: 1 center + 5 ring + 6 outer (symmetric about x-axis)
    inner = [
        [0.0, 0.0, 0.0],
        [x1, y1, 0.0], [-x1, y1, 0.0], [x1, -y1, 0.0], [-x1, -y1, 0.0], [0.0, -2.0 * y1, 0.0],
        [x2, y2, 0.0], [-x2, y2, 0.0], [x2, -y2, 0.0], [-x2, -y2, 0.0],
        [0.0, 2.0 * y2, 0.0], [0.0, -2.0 * y2, 0.0],
    ]
    return np.array(inner, dtype=float)


def _check(inner, outer):
    """Return (s, feasible). s = minimal outer side length for this placement;
    feasible iff all inner-inner separations and all containments hold."""
    cx, cy, ca = outer
    c = np.array([cx, cy])
    verts = [_hex_vertices(*row, 1.0, 0.0) for row in inner]
    s = 0.0
    for v in verts:
        p = _max_vertex_proj(v, c, np.ones((6, 2)))
        s = max(s, p)
    s = 2.0 * s / np.sqrt(3.0)
    feasible = True
    for i in range(12):
        for j in range(i + 1, 12):
            if _separation_gap(verts[i], verts[j]) <= _EPS:
                feasible = False
    n = _hex_vertices(0.0, 0.0, 1.0, 0.0) - np.roll(_hex_vertices(0.0, 0.0, 1.0, 0.0), 1, axis=0)
    n = np.stack([-n[:, 1], n[:, 0]], axis=1)
    for v in verts:
        if _max_vertex_proj(v, c, n) > (np.sqrt(3.0) / 2.0) * s + _CONTAIN_TOL:
            feasible = False
    return s, feasible


def _objective(x):
    inner = _build_initial()
    outer = np.array([x[0], x[1], x[2]])
    s, feasible = _check(inner, outer)
    if not feasible:
        return 1e8 + s
    return s


def _nelder(x0, budget):
    best_x, best_f = x0, _objective(x0)
    options = dict(maxiter=5000, xatol=1e-8, fatol=1e-10, disp=False)
    while (time.time() + _SAFETY) < _DEADLINE:
        try:
            res = _scipy_minimize(_objective, best_x, method="Nelder-Mead",
                                  options=options, tol=1e-10)
        except Exception:
            break
        if res.x is not None and np.all(np.isfinite(res.x)) and res.fun < best_f:
            best_x, best_f = res.x, res.fun
        if best_f < 1e7 and res.fun < 1e7:
            break
    return best_x, best_f


def _refine(inner, outer, s, budget):
    if _scipy_minimize is None:
        return inner, outer, s
    x0 = np.array([outer[0], outer[1], outer[2]])
    best_x, best_f = _nelder(x0, budget)
    if best_f < s and best_f < 1e7:
        outer = np.array([best_x[0], best_x[1], best_x[2]])
        s = best_f
    return inner, outer, s


def hexagon_packing_12():
    inner = _build_initial()
    outer = np.array([0.0, 0.0, 0.0])
    s, feasible = _check(inner, outer)
    if not feasible:
        # Fallback: enlarge the outer hexagon until everything fits (always valid).
        while (time.time() + _SAFETY) < _DEADLINE:
            s, feasible = _check(inner, outer)
            if feasible:
                break
            s *= 1.15
        if not feasible:
            s = 30.0
            feasible = True
    inner, outer, s = _refine(inner, outer, s, _DEADLINE)
    return inner, outer, float(s)


# EVOLVE-BLOCK-END
