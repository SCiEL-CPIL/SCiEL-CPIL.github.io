"""Rewritten packing program: hexagonal-lattice cluster + LP outer hexagon search."""
import numpy as np
from scipy.optimize import linprog, differential_evolution
import time

N = 11
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s limit


def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT


def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon (flat-top when angle=0), CCW."""
    a = np.deg2rad(angle_deg)
    angles = a + np.linspace(0, 2 * np.pi, 7)[:-1]
    return np.stack([cx + side * np.cos(angles), cy + side * np.sin(angles)], axis=1)


def _inner_cluster_config():
    """
    11 unit hexagons in a compact hexagonal arrangement (3-4-3 row pattern).
    Returns list of (x, y, angle_deg) in cluster-local coordinates.
    """
    s = 1.0
    dy = np.sqrt(3) * s  # row spacing for touching hexagons (flat-top)
    dx = 1.5 * s         # horizontal offset within a row
    rows = [(-1, 0, 1), (-2, -1, 0, 1, 2), (-1, 0, 1)]
    # Actually for 3-4-3: rows of 3, 4, 3
    positions = []
    # Row 0 (top): 3 hexagons
    for i in range(-1, 2):
        positions.append((i * 2 * s, 0.0, 0.0))
    # Row 1 (middle): 4 hexagons, shifted
    for i in range(-2, 2):
        positions.append((i * 2 * s, -dy, 0.0))
    # Row 2 (bottom): 3 hexagons
    for i in range(-1, 2):
        positions.append((i * 2 * s, -2 * dy, 0.0))
    # Center the cluster
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(x - cx, y - cy, 0.0) for (x, y, _) in positions]
    return positions


def _compute_outer_side(cluster_pos, theta_cluster, theta_outer, outer_center):
    """
    Compute the minimum side length of a regular hexagon (center outer_center,
    rotation theta_outer) that contains all cluster vertices.
    
    Uses the support function: for a hexagon with side s, the distance from center
    to each of the 6 sides is s * sqrt(3) / 2. So for each vertex, the constraint is:
    dot(vert - center, n_i) <= s * sqrt(3) / 2 for all 6 normals n_i.
    Thus s = max_i dot(vert_i - center, n_i) / (sqrt(3)/2) over all vertices and normals.
    """
    a = np.deg2rad(theta_cluster)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    
    # Transform cluster positions to world
    world_pts = []
    for (px, py, _) in cluster_pos:
        p = np.array([px, py])
        wp = rot @ p + np.array(outer_center)
        # Add vertices of unit hexagon at this position
        verts = _hex_vertices(wp[0], wp[1], 1.0, 0.0)
        world_pts.append(verts)
    world_pts = np.concatenate(world_pts, axis=0)  # (N*6, 2)
    
    # Outer hexagon normals
    b = np.deg2rad(theta_outer)
    normals = np.array([np.cos(b + np.pi * k / 3), np.sin(b + np.pi * k / 3)] for k in range(6))
    normals = np.array(normals)  # (6, 2)
    
    # For each normal, compute max dot product with (world_pts - outer_center)
    rel = world_pts - np.array(outer_center)
    # (N*6, 6) dot products
    dots = rel @ normals.T
    max_dots = np.max(dots, axis=0)  # (6,)
    
    s = np.max(max_dots) / (np.sqrt(3) / 2)
    return max(s, 0.0)


def _objective(params):
    """
    params = [theta_cluster, theta_outer, ox, oy]
    Minimize outer hexagon side length.
    """
    if not _deadline_ok():
        return 1e6
    theta_c = params[0]
    theta_o = params[1]
    ox, oy = params[2], params[3]
    cluster = _inner_cluster_config()
    return _compute_outer_side(cluster, theta_c, theta_o, (ox, oy))


def construct():
    """
    Returns (inner, center, side, angle) where:
      inner: list of [x, y, angle_degrees] for N unit hexagons
      center: (cx, cy) of outer hexagon
      side: side length of outer hexagon
      angle: rotation angle (degrees) of outer hexagon
    """
    global _START_TIME
    _START_TIME = time.time()
    
    cluster_local = _inner_cluster_config()
    
    # Initial best
    best_side = float('inf')
    best_params = None
    
    # Try several initial configurations
    # Since the cluster is symmetric, theta_cluster = 0 is fine; optimize theta_outer and center
    # But let's also try a few theta_cluster values
    
    # Grid search over theta_outer for a quick start
    for tc in [0.0, 30.0, 60.0]:
        for to in range(0, 30, 5):  # 0, 5, 10, 15, 20, 25 degrees
            # Optimize center for fixed angles
            def obj_center(params_c):
                if not _deadline_ok():
                    return 1e6
                ox, oy = params_c
                return _compute_outer_side(cluster_local, tc, to, (ox, oy))
            
            res = differential_evolution(
                obj_center, [(-5, 5), (-5, 5)],
                seed=42, maxiter=50, popsize=15, tol=1e-4
            )
            if res.fun < best_side:
                best_side = res.fun
                best_params = (tc, to, res.x[0], res.x[1])
        
        if not _deadline_ok():
            break
    
    # Refine with full 4D optimization if time permits
    if _deadline_ok() and best_params is not None:
        tc0, to0, ox0, oy0 = best_params
        
        # Use differential_evolution on all 4 params for robustness
        def obj_full(params):
            if not _deadline_ok():
                return 1e6
            tc, to, ox, oy = params
            return _compute_outer_side(cluster_local, tc, to, (ox, oy))
        
        try:
            res = differential_evolution(
                obj_full,
                [(-30, 30), (-30, 30), (-8, 8), (-8, 8)],
                seed=123, maxiter=80, popsize=20, tol=1e-5,
                polish=True
            )
            if res.fun < best_side:
                best_side = res.fun
                best_params = (res.x[0], res.x[1], res.x[2], res.x[3])
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: trivial configuration
        tc, to, ox, oy = 0.0, 0.0, 0.0, 0.0
        best_side = _compute_outer_side(cluster_local, tc, to, (ox, oy))
        best_params = (tc, to, ox, oy)
    
    tc, to, ox, oy = best_params
    
    # Transform cluster positions to world coordinates
    a = np.deg2rad(tc)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    
    inner = []
    for (px, py, ang) in cluster_local:
        p = rot @ np.array([px, py]) + np.array([ox, oy])
        # The individual hexagon angle in world is tc (since local angle is 0)
        inner.append([float(p[0]), float(p[1]), float(tc)])
    
    center = (float(ox), float(oy))
    side = float(best_side)
    angle = float(to)
    
    return inner, center, side, angle
