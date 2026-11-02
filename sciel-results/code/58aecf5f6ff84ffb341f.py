"""
Packing 11 unit regular hexagons inside a regular hexagon.
Strategy: Use a fixed compact lattice configuration (7 center + 4 ring-2).
Optimize the global rotation of the cluster and the orientation of the outer hexagon
using grid search + Nelder-Mead polish and linear programming for the minimal container.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors using convex polygon separation."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    # Quick bounding-circle rejection
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    verts1 = np.array(_hex_vertices(cx1, cy1, s1, a1))
    verts2 = np.array(_hex_vertices(cx2, cy2, s2, a2))
    # Check if any vertex of poly1 is strictly inside poly2
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    # Separating axis theorem: check all 12 edge normals
    for verts, other in [(verts1, verts2), (verts2, verts1)]:
        for i in range(6):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % 6]
            ex = bx - ax
            ey = by - ay
            length = math.sqrt(ex * ex + ey * ey)
            if length < 1e-15:
                continue
            # Normal perpendicular to edge
            nx = -ey / length
            ny = ex / length
            # Project both polygons onto this normal
            proj1 = verts[:, 0] * nx + verts[:, 1] * ny
            proj2 = other[:, 0] * nx + other[:, 1] * ny
            if proj1.max() <= proj2.min() + eps or proj2.max() <= proj1.min() + eps:
                return True
    return False

def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True

def _min_outer_side_analytical(hexes, oangle_rad):
    """
    Find minimum side length of outer hexagon with rotation oangle_rad
    that contains all hexagons in `hexes` (centered at origin).
    Returns (min_side, center_x, center_y)
    """
    # Collect all vertices of all inner hexagons
    points = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        points.extend(verts)
    
    if not points:
        return 0.0, 0.0, 0.0
    
    pts = np.array(points)
    
    # 6 normal directions for the outer hexagon
    thetas = oangle_rad + np.radians(30 + 60 * np.arange(6))
    u_vecs = np.column_stack([np.cos(thetas), np.sin(thetas)])
    
    # For each direction, compute max and min projection
    projections = pts @ u_vecs.T  # shape (n_points, 6)
    P_vals = projections.max(axis=0)  # max projection
    Q_vals = projections.min(axis=0)  # min projection
    
    # Linear programming: minimize A (apothem)
    # Variables: ox, oy, A (3 variables)
    # Constraints:
    # For each k:
    #   ox*u_kx + oy*uky - A <= Q_k
    #   -ox*u_kx - oy*uky - A <= -P_k
    
    A_mat = np.zeros((12, 3))
    b_vec = np.zeros(12)
    for k in range(6):
        ux, uy = u_vecs[k]
        A_mat[2*k, 0] = ux
        A_mat[2*k, 1] = uy
        A_mat[2*k, 2] = -1
        b_vec[2*k] = Q_vals[k]
        
        A_mat[2*k+1, 0] = -ux
        A_mat[2*k+1, 1] = -uy
        A_mat[2*k+1, 2] = -1
        b_vec[2*k+1] = -P_vals[k]
    
    c_vec = np.array([0.0, 0.0, 1.0])
    
    try:
        result = linprog(c_vec, A_ub=A_mat, b_ub=b_vec, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if result.success:
            A_opt = result.x[2]
            ox_opt = result.x[0]
            oy_opt = result.x[1]
            side = A_opt * 2.0 / math.sqrt(3)
            return side, ox_opt, oy_opt
    except Exception:
        pass
    
    # Fallback if linprog fails (should be rare)
    return float('inf'), 0.0, 0.0


def _nm_polish_angle(hexes, theta0_deg, deadline):
    """Polish the outer hexagon angle using Nelder-Mead around theta0_deg."""
    def objective(theta_rad):
        if time.time() > deadline:
            return float('inf')
        side, ox, oy = _min_outer_side_analytical(hexes, theta_rad)
        return side
    
    x0 = np.array([math.radians(theta0_deg)])
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 500})
        if res.success or res.fun < float('inf'):
            best_theta_rad = res.x[0]
            best_side, ox, oy = _min_outer_side_analytical(hexes, best_theta_rad)
            return best_side, best_theta_rad, ox, oy
    except Exception:
        pass
    return float('inf'), theta0_deg, 0.0, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Define 11 positions in axial coordinates (q, r)
    # Center
    pos_axial = [(0, 0)]
    # Ring 1 (6 neighbors)
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    # We need 4 more from Ring 2.
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_xy = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_inner_hexes = None
    best_outer_center = (0.0, 0.0)
    best_combo = None

    # Pre-compute Cartesian coordinates for all potential positions
    all_axial = pos_axial + ring2_candidates
    all_xy = [axial_to_xy(q, r) for q, r in all_axial]
    
    # Indices for ring 2 in all_xy
    ring2_indices = list(range(7, len(all_xy)))

    # Iterate over combinations of 4 ring-2 positions
    for combo in itertools.combinations(ring2_indices, 4):
        if time.time() > deadline:
            break
            
        # Construct the list of 11 points
        selected_indices = list(range(7)) + list(combo)
        full_xy = [all_xy[i] for i in selected_indices]
        
        # Center the cluster at origin for calculation
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        # Shift points to origin
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in full_xy]
        
        # Inner hexagons are aligned with the lattice (angle 0)
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        
        # Fine grid search for coarse angle optimization (0.5 degree steps)
        best_theta_local = 0.0
        best_side_local = float('inf')
        best_ox_local = 0.0
        best_oy_local = 0.0
        
        for t_deg in np.arange(0, 30.001, 0.5):
            if time.time() > deadline:
                break
            t_rad = math.radians(t_deg)
            side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, t_rad)
            if side < best_side_local:
                best_side_local = side
                best_theta_local = t_deg
                best_ox_local = ox
                best_oy_local = oy
        
        # Nelder-Mead polish for high precision
        if time.time() < deadline - 2.0:
            nm_side, nm_theta, nm_ox, nm_oy = _nm_polish_angle(inner_hexes_centered, best_theta_local, deadline)
            if nm_side < best_side_local:
                best_side_local = nm_side
                best_theta_local = math.degrees(nm_theta)
                best_ox_local = nm_ox
                best_oy_local = nm_oy
            
        if best_side_local < best_side:
            best_side = best_side_local
            best_angle = best_theta_local
            best_center = (best_ox_local + cx_mean, best_oy_local + cy_mean)
            best_xy = full_xy
            best_inner_hexes = inner_hexes_centered
            best_outer_center = (best_ox_local, best_oy_local)
            best_combo = combo

    if best_xy is None:
        # Fallback
        best_xy = [all_xy[i] for i in range(7)]
        cx_mean = sum(x for x, y in best_xy) / 7.0
        cy_mean = sum(y for x, y in best_xy) / 7.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in best_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, 0.0)
        best_center = (ox + cx_mean, oy + cy_mean)
        best_angle = 0.0
        best_inner_hexes = inner_hexes_centered
        best_outer_center = (ox, oy)
        best_combo = tuple(range(7, 11))

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        scale = 1.01
        inner_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_xy]
        cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
        cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in [(h[0], h[1]) for h in inner_hexes]]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, math.radians(best_angle))
        best_center = (ox + cx_mean, oy + cy_mean)

    # Final Nelder-Mead polish on the best configuration found
    if time.time() < deadline - 5.0 and best_xy is not None:
        cx_mean = sum(x for x, y in best_xy) / 11.0
        cy_mean = sum(y for x, y in best_xy) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in best_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        
        # Try polish from current angle and a few nearby angles
        for start_angle in [best_angle, best_angle + 0.1, best_angle - 0.1]:
            if time.time() > deadline - 5.0:
                break
            nm_side, nm_theta_rad, nm_ox, nm_oy = _nm_polish_angle(inner_hexes_centered, start_angle, deadline)
            if nm_side < best_side:
                best_side = nm_side
                best_angle = math.degrees(nm_theta_rad)
                best_center = (nm_ox + cx_mean, nm_oy + cy_mean)

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        best_side *= 1.005
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
