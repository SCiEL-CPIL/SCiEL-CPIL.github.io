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

# ---------- honeycomb lattice representation ----------

def _lattice_to_positions(lattice_coords):
    """Convert honeycomb lattice integer coordinates to Cartesian positions.
    
    The honeycomb lattice has two sublattices. We use a basis where:
    - Sublattice A: positions at (i * sqrt(3), j * 3 + i * 1.5) for integers i,j
    - Sublattice B: positions at (i * sqrt(3) + sqrt(3)/2, j * 3 + i * 1.5 + 1.5)
    
    Actually, let's use a simpler representation. The dual of the hexagonal tiling
    is the triangular lattice. Centers of touching unit hexagons are sqrt(3) apart.
    
    For a honeycomb arrangement, we can parameterize by:
    - Which sites are occupied (binary mask)
    - Global rotation and translation
    
    But for optimization, we'll use continuous positions with lattice constraints.
    
    Simpler approach: use the fact that optimal packings often have hexagons on
    a triangular lattice. We parameterize by lattice vectors and which sites are used.
    """
    # This is a placeholder - we'll use direct position optimization but with
    # a better initial guess based on known good packings
    return lattice_coords

def _known_good_packing():
    """Return a known good packing of 11 hexagons.
    
    Based on literature and known results, a good packing has:
    - 1 hexagon in the center
    - 6 hexagons in a ring around it (touching)
    - 4 hexagons in the "corner" positions
    
    The key insight is that the 4 corner hexagons should be placed to minimize
    the bounding hexagon size.
    """
    d = np.sqrt(3.0)
    inner = np.zeros((11, 3))
    
    # Center
    inner[0] = [0.0, 0.0, 0.0]
    
    # Ring of 6
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        inner[k + 1] = [d * np.cos(ang), d * np.sin(ang), 0.0]
    
    # 4 corner positions - these need optimization
    # Initial guess: placed at 45-degree offsets from the ring
    for k in range(4):
        ang = np.deg2rad(30.0 + 90.0 * k)
        inner[7 + k] = [2.0 * d * np.cos(ang), 2.0 * d * np.sin(ang), 0.0]
    
    return inner

def _fast_separation_check(inner):
    """Fast approximate separation check using center distances.
    
    For two unit hexagons with the same orientation, the minimum center distance
    for non-overlap is sqrt(3) (they touch when centers are sqrt(3) apart along
    a normal direction).
    
    This is a necessary but not sufficient condition - we use it as a fast filter.
    """
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            dx = inner[i, 0] - inner[j, 0]
            dy = inner[i, 1] - inner[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < np.sqrt(3.0) - 1e-5:
                return False
    return True

def _objective_lattice(x, outer_angle):
    """Objective for lattice-based optimization.
    
    x contains 22 values: 11 positions (x, y) for each hexagon.
    We fix all angles to 0 for simplicity (hexagons are axis-aligned).
    """
    positions = x.reshape(11, 2)
    inner = np.hstack([positions, np.zeros((11, 1))])
    
    # Fast separation check - return large penalty if violated
    if not _fast_separation_check(inner):
        # Compute actual violation
        pen = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                dx = inner[i, 0] - inner[j, 0]
                dy = inner[i, 1] - inner[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = np.sqrt(3.0)
                if dist < min_dist:
                    pen += (min_dist - dist) ** 2
        return 100.0 + 10.0 * pen
    
    # Compute bounding hexagon side
    oc = np.array([0.0, 0.0])
    side = _outer_side_for(inner, oc, outer_angle)
    return side

def _try_optimize_lattice(inner0, ang, max_iter=200, ftol=1e-10, gtol=1e-8):
    """Optimize positions with lattice-inspired initial guess."""
    try:
        x0 = inner0[:, :2].ravel()
        res = minimize(_objective_lattice, x0, args=(ang,),
                       method="L-BFGS-B", options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol})
        positions = res.x.reshape(11, 2)
        inner2 = np.hstack([positions, np.zeros((11, 1))])
        side2 = _outer_side_for(inner2, np.array([0.0, 0.0]), ang)
        if _validity(inner2, np.array([0.0, 0.0]), ang, side2):
            return inner2, side2
    except Exception:
        pass
    return None, None

def hexagon_packing_11():
    """Find optimal packing of 11 unit hexagons in smallest regular hexagon."""
    oc0 = np.array([0.0, 0.0])
    ang0 = 0.0
    
    # Start with known good packing
    inner0 = _known_good_packing()
    s0 = _outer_side_for(inner0, oc0, ang0)
    
    if _validity(inner0, oc0, ang0, s0):
        best_inner = inner0.copy()
        best_side = s0
    else:
        # Fallback: spread out
        inner0 = np.array([
            [0, 0, 0], [-3.0, 0, 0], [3.0, 0, 0], [-1.5, 2.6, 0], [1.5, 2.6, 0],
            [-1.5, -2.6, 0], [1.5, -2.6, 0], [-4.5, 2.6, 0], [4.5, 2.6, 0],
            [-4.5, -2.6, 0], [4.5, -2.6, 0],
        ], dtype=float)
        s0 = _outer_side_for(inner0, oc0, ang0)
        best_inner = inner0.copy()
        best_side = s0

    # Optimization passes with time budget
    elapsed = time.time() - _START_TIME
    if elapsed < _TIME_LIMIT - 10.0:
        # Pass 1: L-BFGS-B with lattice-inspired initialization
        inner_opt, side_opt = _try_optimize_lattice(best_inner, ang0, max_iter=150, ftol=1e-10, gtol=1e-8)
        if inner_opt is not None and side_opt < best_side:
            best_inner = inner_opt
            best_side = side_opt

    # Pass 2: Refine with more iterations if time permits
    elapsed = time.time() - _START_TIME
    if elapsed < _TIME_LIMIT - 30.0 and best_side > np.sqrt(11.0) * 1.02:
        inner_opt2, side_opt2 = _try_optimize_lattice(best_inner, ang0, max_iter=300, ftol=1e-11, gtol=1e-9)
        if inner_opt2 is not None and side_opt2 < best_side:
            best_inner = inner_opt2
            best_side = side_opt2

    # Try different outer angles
    elapsed = time.time() - _START_TIME
    if elapsed < _TIME_LIMIT - 20.0:
        for trial_ang in [30.0, 15.0, -15.0, 45.0, -45.0]:
            if time.time() - _START_TIME > _TIME_LIMIT - 15.0:
                break
            s_trial = _outer_side_for(best_inner, oc0, trial_ang)
            if s_trial < best_side:
                if _validity(best_inner, oc0, trial_ang, s_trial):
                    best_side = s_trial
                    ang0 = trial_ang

    return best_inner, np.array([oc0[0], oc0[1], ang0]), float(best_side)

# EVOLVE-BLOCK-END
