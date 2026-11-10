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
        norm = np.linalg.norm(normal)
        if norm < 1e-15:
            normal = np.array([1.0, 0.0])
        else:
            normal /= norm
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
    inner vertices. Uses the support function."""
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
    """Sum of squared penetration depths over all pairs.
    Uses a fast distance-based approximation for speed during optimization."""
    n = len(inner)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = inner[i, 0] - inner[j, 0]
            dy = inner[i, 1] - inner[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = np.sqrt(3.0)
            if dist < min_dist:
                pen += (min_dist - dist) ** 2
    return pen

def _objective(x, outer_center, outer_angle):
    inner = x.reshape(11, 3)
    return _outer_side_for(inner, outer_center, outer_angle) + 10.0 * _penalty(inner)

def _initial_packing():
    """7 hexagons in a hexagonal lattice (center + ring) + 4 in corner gaps.
    Centers of touching unit hexagons are sqrt(3) apart."""
    d = np.sqrt(3.0)
    inner = np.zeros((11, 3))
    inner[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        inner[k + 1] = [d * np.cos(np.deg2rad(60.0 * k)), d * np.sin(np.deg2rad(60.0 * k)), 0.0]
    # corner-gap positions
    for k in range(4):
        ang = np.deg2rad(30.0 + 90.0 * k)
        inner[7 + k] = [2.0 * d * np.cos(ang), 2.0 * d * np.sin(ang), 0.0]
    return inner

def _try_optimize(inner0, oc, ang, max_iter=50, ftol=1e-8, gtol=1e-6):
    """Attempt a single L-BFGS-B optimization and return improved result if valid."""
    try:
        x0 = inner0.ravel()
        res = minimize(_objective, x0, args=(oc, ang),
                       method="L-BFGS-B", options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol})
        inner2 = res.x.reshape(11, 3)
        side2 = _outer_side_for(inner2, oc, ang)
        if _validity(inner2, oc, ang, side2):
            return inner2, side2
    except Exception:
        pass
    return None, None

def hexagon_packing_11():
    # 1. Generate initial packing
    inner0 = _initial_packing()
    oc0 = np.array([0.0, 0.0])
    ang0 = 0.0
    
    # Check if initial is valid
    s0 = _outer_side_for(inner0, oc0, ang0)
    if _validity(inner0, oc0, ang0, s0):
        best_inner = inner0.copy()
        best_side = s0
    else:
        # Fallback to a very spread out packing to ensure validity
        inner0 = np.array([
            [0, 0, 0], [-3.0, 0, 0], [3.0, 0, 0], [-1.5, 2.6, 0], [1.5, 2.6, 0],
            [-1.5, -2.6, 0], [1.5, -2.6, 0], [-4.5, 2.6, 0], [4.5, 2.6, 0],
            [-4.5, -2.6, 0], [4.5, -2.6, 0],
        ], dtype=float)
        s0 = _outer_side_for(inner0, oc0, ang0)
        best_inner = inner0.copy()
        best_side = s0

    # 2. Try optimization with time budget
    if time.time() - _START_TIME < _TIME_LIMIT - 10.0:
        # First pass: moderate iterations
        inner_opt, side_opt = _try_optimize(best_inner, oc0, ang0, max_iter=100, ftol=1e-9, gtol=1e-7)
        if inner_opt is not None and side_opt < best_side:
            best_inner = inner_opt
            best_side = side_opt

        # Second pass: refine further if time permits
        if time.time() - _START_TIME < _TIME_LIMIT - 30.0 and best_side > np.sqrt(11.0) * 1.05:
            inner_opt2, side_opt2 = _try_optimize(best_inner, oc0, ang0, max_iter=200, ftol=1e-10, gtol=1e-8)
            if inner_opt2 is not None and side_opt2 < best_side:
                best_inner = inner_opt2
                best_side = side_opt2

    # 3. Try different outer angles to potentially reduce the enclosing hexagon size
    if time.time() - _START_TIME < _TIME_LIMIT - 20.0:
        for trial_ang in [30.0, 15.0, -15.0]:
            if time.time() - _START_TIME > _TIME_LIMIT - 15.0:
                break
            s_trial = _outer_side_for(best_inner, oc0, trial_ang)
            if s_trial < best_side:
                # Check validity with new angle
                if _validity(best_inner, oc0, trial_ang, s_trial):
                    best_side = s_trial
                    ang0 = trial_ang

    return best_inner, np.array([oc0[0], oc0[1], ang0]), float(best_side)

# EVOLVE-BLOCK-END
