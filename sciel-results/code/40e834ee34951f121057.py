"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Combines lattice enumeration (current) with LP-based outer hexagon optimization (inspiration).
Uses scipy.optimize.linprog for exact outer hexagon sizing, plus nonlinear local polish.
"""
import math
import time
import itertools
import numpy as np

N = 11
DEADLINE = 115.0  # safety margin before 150s limit


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


def _outer_hexagon_lp(inner_hexes, oangle=0.0):
    """Use LP to find optimal center and side of outer hexagon containing all inner hexagons.
    Returns (cx, cy, side)."""
    from scipy.optimize import linprog
    
    # Variables: [cx, cy, t] where t is apothem of outer hex
    # Minimize t
    # Constraints: for each vertex v of each inner hex, for each of 6 normals of outer hex:
    #   (vx - cx) * nx_k + (vy - cy) * ny_k <= t
    #   => -nx_k * cx - ny_k * cy - t <= -vx * nx_k - vy * ny_k
    c = [0, 0, 1]
    A_ub = []
    b_ub = []
    
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                theta = math.radians(oangle + 30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    
    # t >= 0
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx = res.x[0]
            cy = res.x[1]
            t = res.x[2]
            side = t * 2.0 / math.sqrt(3)
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback
    cx_mean = sum(h[0] for h in inner_hexes) / len(inner_hexes)
    cy_mean = sum(h[1] for h in inner_hexes) / len(inner_hexes)
    side = _min_outer_side(inner_hexes, cx_mean, cy_mean, oangle)
    return cx_mean, cy_mean, side


def _local_polish(inner_hexes, oangle, init_cx, init_cy, max_iter=50):
    """Local coordinate-descent polish of outer hexagon center and rotation."""
    cx, cy = init_cx, init_cy
    side = _min_outer_side(inner_hexes, cx, cy, oangle)
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        improved = False
        # Try small shifts in center
        for step in [0.02, 0.01, 0.005, 0.001]:
            for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                           (step,step),(-step,step),(step,-step),(-step,-step)]:
                new_side = _min_outer_side(inner_hexes, cx+dx, cy+dy, oangle)
                if new_side < side - 1e-13:
                    cx, cy, side = cx+dx, cy+dy, new_side
                    improved = True
                    break
            if improved:
                break
        # Try small rotation changes
        for step in [0.5, 0.25, 0.1, 0.05]:
            for da in [step, -step]:
                new_side = _min_outer_side(inner_hexes, cx, cy, oangle+da)
                if new_side < side - 1e-13:
                    oangle, side = oangle+da, new_side
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return cx, cy, oangle, side


def construct():
    start_time = time.time()
    
    sqrt3 = math.sqrt(3)
    
    # Flat-top hexagon mapping using axial coordinates
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)
    
    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (14 hexagons)
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Outer hexagon rotation angles to try
    outer_angles = [0.0, 30.0, 15.0, 45.0, -15.0, -30.0]
    
    # Enumerate all C(14,4) = 1001 combinations of ring-2 hexagons
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() - start_time > DEADLINE:
            break
        
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Build inner hexagons (all at angle 0 initially)
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Quick disjointness check (lattice positions should be disjoint)
        if not _all_disjoint(inner_hexes):
            continue
        
        # Try each outer hexagon rotation
        for oangle in outer_angles:
            if time.time() - start_time > DEADLINE:
                break
            
            # Use LP to get optimal center and side
            cx, cy, side = _outer_hexagon_lp(inner_hexes, oangle)
            
            # Local polish
            cx, cy, oangle, side = _local_polish(inner_hexes, oangle, cx, cy, max_iter=30)
            
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (cx, cy)
                best_angle = oangle
    
    # Fallback if nothing found
    if best_positions is None:
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        best_center, best_side = _outer_hexagon_lp(inner_hexes, 0.0)[:2], _min_outer_side(inner_hexes, 0, 0, 0)
        best_angle = 0.0
    
    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
    
    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Scale up positions slightly
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_center, best_side = _outer_hexagon_lp(inner_hexes, best_angle)[:2], _min_outer_side(inner_hexes, cx_mean, cy_mean, best_angle)
                break
    
    # Verify containment and adjust if needed
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_angle)
            break
    
    # Final check
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            best_side *= 1.005
            outer_hex = (best_center[0], best_center[1], best_side, best_angle)
            break
    
    # Format output
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    
    return inner, (best_center[0], best_center[1]), best_side, best_angle
