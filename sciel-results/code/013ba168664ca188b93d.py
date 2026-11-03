"""Direct hexagon packing: 11 unit hexagons in a hexagonal lattice pattern with LP-optimized outer hexagon."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog

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


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True


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


def _min_outer_side_lp(inner_hexes):
    """Find minimum side length of outer regular hexagon that contains all inner hexagons,
    optimizing center (ox, oy) and angle simultaneously using LP.
    
    The outer hexagon is defined by 6 half-planes. For a regular hexagon with side s,
    apothem = s * sqrt(3) / 2. The constraint is that for each of the 6 face normals n_k,
    the projection of every vertex onto n_k must be <= apothem (in the hexagon's local frame).
    
    We parameterize: outer center (ox, oy), angle theta, side s.
    For each face k (k=0..5), normal direction in world frame is at angle theta + 30 + 60k.
    Constraint: for all vertices v of all inner hexagons,
        (v - center) · n_k <= s * sqrt(3) / 2
    
    This is linear in (ox, oy, s) for fixed theta. We can do LP over (ox, oy, s) for each theta,
    then optimize theta.
    
    Actually, let's use a simpler approach: for a given set of vertices, find the minimum
    enclosing regular hexagon by optimizing (ox, oy, theta) and computing s from the max projection.
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for cx, cy, s, ang in inner_hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.extend(verts)
    
    if not all_verts:
        return 0.0, (0.0, 0.0), 0.0
    
    all_verts = np.array(all_verts)
    n_verts = len(all_verts)
    
    # For a regular hexagon with angle theta, the 6 face normals are at angles theta + 30 + 60k
    # The constraint is: max over all vertices v of (v - center) · n_k <= s * sqrt(3)/2 for all k
    # So s >= max_k [max_v (v - center) · n_k] * 2 / sqrt(3)
    # We want to minimize this over center and theta.
    
    # For fixed theta, the problem is: minimize max_k max_v [(v - c) · n_k] over c
    # This is equivalent to: minimize t such that (v - c) · n_k <= t for all v, k
    # Which is: v · n_k - c · n_k <= t for all v, k
    # => c · n_k + t >= v · n_k for all v, k
    # => -c · n_k - t <= -v · n_k for all v, k
    
    # LP variables: c_x, c_y, t (3 variables)
    # Constraints: for each vertex v and each face k:
    #   -c_x * n_k_x - c_y * n_k_y - t <= -v_x * n_k_x - v_y * n_k_y
    # Minimize t
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_theta = 0.0
    
    # Try a range of theta values
    for theta_deg in range(0, 180, 5):
        theta = math.radians(theta_deg)
        
        # 6 face normals
        normals = []
        for k in range(6):
            angle = theta + math.radians(30 + 60 * k)
            normals.append((math.cos(angle), math.sin(angle)))
        
        # Build LP: minimize t
        # Variables: [cx, cy, t]
        # Objective: [0, 0, 1]
        c_obj = np.array([0.0, 0.0, 1.0])
        
        # Constraints: -cx * nx - cy * ny - t <= -vx * nx - vy * ny
        # => A_ub @ [cx, cy, t] <= b_ub
        A_ub = []
        b_ub = []
        
        for vx, vy in all_verts:
            for nx, ny in normals:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-vx * nx - vy * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
            if res.success:
                cx = res.x[0]
                cy = res.x[1]
                t = res.x[2]
                side = t * 2.0 / math.sqrt(3)
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_theta = theta_deg
        except:
            pass
    
    # Refine around the best theta with finer steps
    if best_side < float('inf'):
        for theta_deg in np.arange(best_theta - 5, best_theta + 5.5, 0.5):
            theta = math.radians(theta_deg)
            
            normals = []
            for k in range(6):
                angle = theta + math.radians(30 + 60 * k)
                normals.append((math.cos(angle), math.sin(angle)))
            
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            
            for vx, vy in all_verts:
                for nx, ny in normals:
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-vx * nx - vy * ny)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
                if res.success:
                    cx = res.x[0]
                    cy = res.x[1]
                    t = res.x[2]
                    side = t * 2.0 / math.sqrt(3)
                    if side < best_side:
                        best_side = side
                        best_center = (cx, cy)
                        best_theta = theta_deg
            except:
                pass
    
    return best_side, best_center, best_theta


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping using axial coordinates:
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try all combinations of 4 from ring2
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Use LP to find optimal outer hexagon
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        side, center, angle = _min_outer_side_lp(inner_hexes)
        
        if side < best_side:
            best_side = side
            best_positions = full_xy
            best_center = center
            best_angle = angle

    if best_positions is None:
        # Fallback
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        best_side, best_center, best_angle = _min_outer_side_lp(inner_hexes)

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_side, best_center, best_angle = _min_outer_side_lp(scaled_hexes)
                best_positions = [(x, y) for x, y, s, a in scaled_hexes]
                break

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        best_side *= 1.01

    # Format output
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
