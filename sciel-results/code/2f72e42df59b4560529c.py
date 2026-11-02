"""Direct hexagon packing: 11 unit hexagons with LP-optimized outer hexagon and inner rotation search."""
import math
import time
import itertools
import numpy as np

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


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
    that contains all hexagons in `hexes`."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)


def _lp_min_enclosing_hex(vertices, oangle=0.0):
    """Use LP to find the minimum side length and center of a regular hexagon
    with fixed rotation oangle that encloses all given vertices.
    
    A regular hexagon with center (cx, cy), side s, and rotation oangle can be
    described by 6 half-plane constraints. For each normal direction n_k,
    the constraint is: n_k . (v - c) <= s * sqrt(3)/2 for all vertices v.
    
    We want to minimize s subject to these constraints.
    Variables: cx, cy, s
    Constraints: for each vertex v and each direction k:
        n_kx * (vx - cx) + n_ky * (vy - cy) <= s * sqrt(3)/2
        i.e., n_kx * vx + n_ky * vy - n_kx * cx - n_ky * cy - s*sqrt(3)/2 <= 0
    
    This is a linear program in variables (cx, cy, s) with objective minimize s.
    """
    from scipy.optimize import linprog
    
    sqrt3 = math.sqrt(3)
    apothem_factor = sqrt3 / 2.0  # s * sqrt(3)/2
    
    # Precompute normal directions
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Rotate normals by -oangle to get them in the fixed coordinate system
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    rotated_normals = []
    for nx, ny in normals:
        rx = nx * cos_a - ny * sin_a
        ry = nx * sin_a + ny * cos_a
        rotated_normals.append((rx, ry))
    
    n_vertices = len(vertices)
    n_constraints = n_vertices * 6
    
    # Objective: minimize s -> c = [0, 0, 1]
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Constraints: A_ub @ [cx, cy, s] <= b_ub
    # For each vertex v=(vx,vy) and each normal n=(nx,ny):
    #   nx*(vx - cx) + ny*(vy - cy) <= s * sqrt(3)/2
    #   -nx*cx - ny*cy - sqrt(3)/2 * s <= -nx*vx - ny*vy
    A_ub = np.zeros((n_constraints, 3))
    b_ub = np.zeros(n_constraints)
    
    idx = 0
    for vx, vy in vertices:
        for nx, ny in rotated_normals:
            A_ub[idx, 0] = -nx
            A_ub[idx, 1] = -ny
            A_ub[idx, 2] = -apothem_factor
            b_ub[idx] = -(nx * vx + ny * vy)
            idx += 1
    
    # s must be positive
    bounds = [(None, None), (None, None), (0.0, None)]
    
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.status == 0:
        cx, cy, s = result.x
        return cx, cy, s
    else:
        # Fallback: compute analytically
        return _min_outer_side_fallback(vertices, oangle)


def _min_outer_side_fallback(vertices, oangle=0.0):
    """Fallback: compute min enclosing hexagon side by trying centroid as center."""
    if not vertices:
        return 0.0, 0.0, 0.0
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    sqrt3 = math.sqrt(3)
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    max_val = 0.0
    for vx, vy in vertices:
        dx = vx - cx
        dy = vy - cy
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            proj = lx * math.cos(theta) + ly * math.sin(theta)
            if proj > max_val:
                max_val = proj
    side = max_val * 2.0 / sqrt3
    return cx, cy, side


def _optimize_outer_lp(inner_hexes, initial_oangle=0.0, max_angle_iters=10):
    """Optimize outer hexagon using LP for center+size, and iterate on angle."""
    sqrt3 = math.sqrt(3)
    
    # Collect all vertices
    all_verts = []
    for cx, cy, s, ang in inner_hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.extend(verts)
    
    best_side = float('inf')
    best_cx, best_cy, best_angle = 0.0, 0.0, initial_oangle
    
    # Try several initial angles
    initial_angles = [initial_oangle, initial_oangle + 30.0, 0.0, 30.0, 60.0, 15.0, 45.0]
    
    for oang in initial_angles:
        cx, cy, s = _lp_min_enclosing_hex(all_verts, oang)
        if s < best_side:
            best_side = s
            best_cx, best_cy, best_angle = cx, cy, oang
    
    # Refine angle around best
    for _ in range(max_angle_iters):
        improved = False
        for da in [5.0, -5.0, 2.0, -2.0, 1.0, -1.0, 0.5, -0.5]:
            new_ang = best_angle + da
            cx, cy, s = _lp_min_enclosing_hex(all_verts, new_ang)
            if s < best_side - 1e-10:
                best_side = s
                best_cx, best_cy, best_angle = cx, cy, new_ang
                improved = True
        if not improved:
            break
    
    return best_cx, best_cy, best_angle, best_side


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping using axial coordinates:
    # For a unit hexagon (side=1) with flat-top orientation:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # We need to place 11 hexagons. Start with a hexagonal lattice of 7 (center + 6 neighbors),
    # then add 4 more from ring 2.
    
    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (distance 2 from center) - 12 hexagons
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_inner_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try all combinations of 4 from ring2
    # C(14, 4) = 1001 combinations
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline - 5.0:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Standard configuration: all inner hexagons at 0 degrees
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Use LP to find optimal outer hexagon
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        opt_cx, opt_cy, opt_angle, opt_side = _optimize_outer_lp(
            inner_hexes, initial_oangle=0.0, max_angle_iters=5
        )
        
        if opt_side < best_side:
            best_side = opt_side
            best_positions = full_xy
            best_inner_angles = [0.0] * 11
            best_center = (opt_cx, opt_cy)
            best_angle = opt_angle

    # Now try some rotated inner configurations for the best position set
    if best_positions is not None:
        # Try rotating individual hexagons to see if we can get a tighter packing
        # Focus on the 4 ring-2 hexagons (indices 7-10)
        base_positions = list(best_positions)
        
        # Try rotating ring-2 hexagons by 30 degrees (pointy-top vs flat-top)
        rotation_options = [0.0, 30.0]
        
        for r2_rot in itertools.product(rotation_options, repeat=4):
            if time.time() > deadline - 3.0:
                break
            
            test_positions = list(base_positions)
            test_angles = [0.0] * 11
            for i, rot in enumerate(r2_rot):
                test_angles[7 + i] = rot
            
            inner_hexes = [(test_positions[i][0], test_positions[i][1], 1.0, test_angles[i]) 
                          for i in range(11)]
            
            # Check disjointness
            if not _all_disjoint(inner_hexes):
                continue
            
            # Use LP to find optimal outer hexagon
            opt_cx, opt_cy, opt_angle, opt_side = _optimize_outer_lp(
                inner_hexes, initial_oangle=best_angle, max_angle_iters=3
            )
            
            if opt_side < best_side:
                best_side = opt_side
                best_inner_angles = test_angles
                best_center = (opt_cx, opt_cy)
                best_angle = opt_angle

    if best_positions is None:
        # Fallback: just use the 7-hexagon cluster
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        best_inner_angles = [0.0] * 7
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        cx_mean = sum(x for x, y in best_positions) / 7.0
        cy_mean = sum(y for x, y in best_positions) / 7.0
        best_cx, best_cy, best_angle, best_side = _optimize_outer_lp(
            inner_hexes, initial_oangle=0.0, max_angle_iters=5
        )
        best_center = (best_cx, best_cy)

    # Build final inner hexagons
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_inner_angles[i]) 
                  for i in range(len(best_positions))]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # If there's an issue, try scaling up positions slightly
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / len(inner_hexes)
                cy_mean = sum(y for x, y, s, a in inner_hexes) / len(inner_hexes)
                best_cx, best_cy, best_angle, best_side = _optimize_outer_lp(
                    inner_hexes, initial_oangle=0.0, max_angle_iters=5
                )
                best_center = (best_cx, best_cy)
                break

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side slightly to ensure containment
        best_side *= 1.01
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
