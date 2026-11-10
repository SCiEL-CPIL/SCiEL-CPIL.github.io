# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # safety margin before the 630s wall-clock kill

# ---------- geometry helpers ----------

def _hex_vertices(cx, cy, s, a_deg):
    """(6,2) vertex array of a regular hexagon (center, side, angle_deg)."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _edge_normals(vertices):
    """(6,2) outward-pointing unit edge normals of a convex polygon."""
    n = len(vertices)
    centroid = vertices.mean(axis=0)
    norms = np.empty((n, 2))
    for i in range(n):
        edge = vertices[(i + 1) % n] - vertices[i]
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)
        mid = 0.5 * (vertices[i] + vertices[(i + 1) % n])
        if np.dot(normal, mid - centroid) < 0.0:
            normal = -normal
        norms[i] = normal
    return norms


def _min_gap(v1, v2, n1, n2):
    """Minimum separation of two convex hexagons via SAT (12 edge normals).
    Positive => separated, 0 => touching/overlapping."""
    best = np.inf
    for n in np.vstack((n1, n2)):
        a = v1 @ n
        b = v2 @ n
        if a.max() < b.min():
            g = b.min() - a.max()
        elif b.max() < a.min():
            g = a.min() - b.max()
        else:
            g = 0.0
        if g < best:
            best = g
    return best


def _separation(h1, h2):
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    return _min_gap(v1, v2, _edge_normals(v1), _edge_normals(v2))


def _outer_side_for(inner, outer_center, outer_angle):
    """Smallest side of a regular hexagon (center, angle) that encloses all
    inner vertices.  Uses the support function: s >= 2/sqrt(3) * max_k max_p (p·n_k)."""
    a = np.deg2rad(outer_angle)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))  # outward normal directions
    normals = np.column_stack([np.cos(th), np.sin(th)])
    verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) - outer_center for h in inner])
    max_proj = (verts @ normals.T).max(axis=0)
    return float((2.0 / np.sqrt(3.0)) * max_proj.max())


def _validity(inner, outer_center, outer_angle, side):
    """Return True if the packing is valid (separated + contained + finite)."""
    if not np.all(np.isfinite(inner)):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    verts = [_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner]
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            if _min_gap(verts[i], verts[j], _edge_normals(verts[i]), _edge_normals(verts[j])) <= 1e-6:
                return False
    ov = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    for v in verts:
        for k in range(6):
            p1, p2 = ov[k], ov[(k + 1) % 6]
            if np.cross(p2 - p1, v - p1) < -1e-6:
                return False
    return True


# ---------- search machinery ----------

def _penalty(inner):
    """Sum of squared penetration depths over all pairs (0 if fully separated)."""
    n = len(inner)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            g = _separation(inner[i], inner[j])
            if g < 1e-6:
                pen += (1e-6 - g) ** 2
    return pen


def _objective(x, outer_center, outer_angle):
    inner = x.reshape(11, 3)
    return _outer_side_for(inner, outer_center, outer_angle) + 100.0 * _penalty(inner)


def _initial_packing():
    """7 hexagons in a hexagonal lattice (center + ring) + 4 in corner gaps.
    Centers of touching unit hexagons are sqrt(3) apart."""
    d = np.sqrt(3.0)
    inner = np.zeros((11, 3))
    inner[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        inner[k + 1] = [d * np.cos(np.deg2rad(60.0 * k)), d * np.sin(np.deg2rad(60.0 * k)), 0.0]
    # corner-gap positions: equilateral triangle third vertex, distance 2d at 30+60k
    for k in range(4):
        ang = np.deg2rad(30.0 + 90.0 * k)  # 30, 120, 210, 300
        inner[7 + k] = [2.0 * d * np.cos(ang), 2.0 * d * np.sin(ang), 0.0]
    return inner


def _try_refine(inner, outer_center, outer_angle, best, tag):
    global _best_side
    x0 = inner.ravel()
    res = minimize(_objective, x0, args=(outer_center, outer_angle),
                   method="L-BFGS-B", options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-8})
    inner2 = res.x.reshape(11, 3)
    side2 = _outer_side_for(inner2, outer_center, outer_angle)
    if _validity(inner2, outer_center, outer_angle, side2) and side2 < _best_side:
        _best_side = side2
        best = (inner2.copy(), outer_center.copy(), outer_angle, side2)
    return best


def hexagon_packing_11():
    global _best_side
    _best_side = np.inf

    # Symmetric seed: 1 center + 6 ring + 4 corners.
    inner0 = _initial_packing()
    oc0 = np.array([0.0, 0.0])
    best = None
    s0 = _outer_side_for(inner0, oc0, 0.0)
    if _validity(inner0, oc0, 0.0, s0):
        _best_side = s0
        best = (inner0.copy(), oc0.copy(), 0.0, s0)
    else:  # fall back to a guaranteed-valid trivial packing
        inner0 = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0], [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ], dtype=float)
        s0 = _outer_side_for(inner0, oc0, 0.0)
        _best_side = s0
        best = (inner0.copy(), oc0.copy(), 0.0, s0)

    # Outer-center search (symmetric packing => center at origin is optimal,
    # but a light scan guards against asymmetry introduced by the optimizer).
    for dx in (0.0, 0.1, -0.1, 0.2, -0.2):
        for dy in (0.0, 0.1, -0.1, 0.2, -0.2):
            if time.time() - _START_TIME > _TIME_LIMIT:
                break
            oc = np.array([dx, dy])
            side = _outer_side_for(best[0], oc, best[2])
            if _validity(best[0], oc, best[2], side) and side < _best_side:
                _best_side = side
                best = (best[0].copy(), oc.copy(), best[2], side)

    # Outer-angle search.
    for ang in (0.0, 15.0, 30.0, 45.0, 7.5, 22.5):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        side = _outer_side_for(best[0], best[1], ang)
        if _validity(best[0], best[1], ang, side) and side < _best_side:
            _best_side = side
            best = (best[0].copy(), best[1].copy(), ang, side)

    # Local refinement of inner positions, with a few randomized restarts.
    if time.time() - _START_TIME < _TIME_LIMIT:
        best = _try_refine(best[0], best[1], best[2], best, "base")

    rng = np.random.default_rng(0)
    for _ in range(40):
        if time.time() - _START_TIME > _TIME_LIMIT - 40.0:
            break
        pert = inner0 + rng.normal(0.0, 0.15, size=(11, 3))
        pert[:, 2] = rng.uniform(-15.0, 15.0, size=11)
        best = _try_refine(pert, best[1], best[2], best, "restart")

    inner, oc, ang, side = best
    return inner, np.array([oc[0], oc[1], ang]), float(side)


# EVOLVE-BLOCK-END
