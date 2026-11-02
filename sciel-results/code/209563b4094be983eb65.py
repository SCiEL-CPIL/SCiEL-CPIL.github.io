"""Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses a hexagonal lattice pattern with LP-based outer hexagon optimization
and local search for tighter packing."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 11


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point is inside hexagon."""
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
    """Check if two segments properly intersect."""
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


def _support_function(point, direction):
    """Dot product of point with direction vector."""
    return point[0] * direction[0] + point[1] * direction[1]


def _min_outer_side_lp(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon using LP formulation.
    The outer hexagon must contain all vertices of all inner hexagons.
    For a regular hexagon with center (ox, oy), side s, rotation oangle:
    The constraint is that for each vertex v of each inner hex,
    the projection onto each of the 6 normal directions must be <= s*sqrt(3)/2.
    We want to minimize s subject to these constraints.
    Since the constraints are linear in s (and the geometry is fixed once
    center and angle are fixed), this is just finding the max projection.
    """
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


def _compute_outer_for_positions(positions, angles=None):
    """Given inner hexagon centers (and optional angles), find optimal outer hexagon.
    Uses scipy minimize to optimize center and angle of outer hexagon."""
    if angles is None:
        angles = [0.0] * len(positions)
    
    inner_hexes = [(x, y, 1.0, a) for (x, y), a in zip(positions, angles)]
    
    def objective(params):
        ox, oy, oang = params
        return _min_outer_side_lp(inner_hexes, ox, oy, oang)
    
    # Multi-start optimization
    best = float('inf')
    best_params = None
    
    # Centroid as initial guess
    cx0 = sum(p[0] for p in positions) / len(positions)
    cy0 = sum(p[1] for p in positions) / len(positions)
    
    # Try multiple starting angles
    for oang0 in [0.0, 30.0, 15.0, -15.0, 45.0, -45.0]:
        x0 = [cx0, cy0, oang0]
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best:
                best = res.fun
                best_params = res.x
        except:
            pass
    
    if best_params is None:
        return cx0, cy0, 0.0, _min_outer_side_lp(inner_hexes, cx0, cy0, 0.0)
    
    return best_params[0], best_params[1], best_params[2], best


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping using axial coordinates
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Center + ring 1 (6 hexagons) = 7 hexagons
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (distance 2 from center) - 12 hexagons
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Pre-compute all ring2 combinations of 4
    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    
    # Phase 1: Scan combinations with LP-based outer optimization
    for combo in all_combos:
        if time.time() > deadline - 60.0:  # Leave 60s for phase 2
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Use LP-based outer hexagon optimization
        ox, oy, oang, side = _compute_outer_for_positions(full_xy)
        
        if side < best_side:
            best_side = side
            best_positions = full_xy
            best_angles = [0.0] * 11
            best_center = (ox, oy)
            best_angle = oang
    
    # Phase 2: Local optimization of inner hexagon positions
    if best_positions is not None and time.time() < deadline - 30.0:
        # Optimize positions of the 4 ring-2 hexagons while keeping ring 0 and 1 fixed
        # Use scipy minimize to reduce the outer side length
        
        def outer_side_from_positions(params):
            """Given positions of 4 ring-2 hexagons, compute outer side length."""
            # First 7 positions are fixed
            fixed = best_positions[:7]
            # Last 4 are variable
            var = [(params[i*2], params[i*2+1]) for i in range(4)]
            all_pos = fixed + var
            _, _, _, side = _compute_outer_for_positions(all_pos)
            return side
        
        x0 = [coord for pos in best_positions[7:] for coord in pos]
        
        try:
            res = minimize(outer_side_from_positions, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_side:
                new_var = [(res.x[i*2], res.x[i*2+1]) for i in range(4)]
                new_positions = best_positions[:7] + new_var
                ox, oy, oang, side = _compute_outer_for_positions(new_positions)
                if side < best_side:
                    best_side = side
                    best_positions = new_positions
                    best_center = (ox, oy)
                    best_angle = oang
        except:
            pass
    
    # Phase 3: Fine-tune outer hexagon parameters
    if best_positions is not None and time.time() < deadline - 10.0:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        
        # High-precision optimization of outer hexagon
        def obj(params):
            return _min_outer_side_lp(inner_hexes, params[0], params[1], params[2])
        
        x0 = [best_center[0], best_center[1], best_angle]
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_side:
                best_side = res.fun
                best_center = (res.x[0], res.x[1])
                best_angle = res.x[2]
        except:
            pass
    
    # Fallback if no valid configuration found
    if best_positions is None:
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial[:7]]
        best_angles = [0.0] * 7
        best_side = _min_outer_side_lp([(x, y, 1.0, 0.0) for x, y in best_positions], 0, 0, 0)
        best_center = (0, 0)
        best_angle = 0.0

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, a) for (x, y), a in zip(best_positions, best_angles)]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Try scaling up positions slightly
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for (x, y, s, a) in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / len(inner_hexes)
                cy_mean = sum(y for x, y, s, a in inner_hexes) / len(inner_hexes)
                best_side = _min_outer_side_lp(inner_hexes, cx_mean, cy_mean, 0.0)
                best_center = (cx_mean, cy_mean)
                best_angle = 0.0
                break

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Binary search for minimum valid side length
        lo, hi = best_side, best_side * 1.1
        for _ in range(50):
            mid = (lo + hi) / 2
            test_outer = (best_center[0], best_center[1], mid, best_angle)
            ok = all(_inner_in_outer(h, test_outer) for h in inner_hexes)
            if ok:
                hi = mid
            else:
                lo = mid
        best_side = hi
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
