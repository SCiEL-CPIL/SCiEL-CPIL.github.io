"""Pack 11 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE_MARGIN = 5.0  # seconds before the 150s wall-clock limit
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _hex_vertices(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon (flat-top if ang=0)."""
    a = math.radians(ang_deg)
    # Regular hexagon vertices at angles a + k*60 degrees, radius = s
    # For a regular hexagon with side length s, circumradius = s
    verts = []
    for k in range(6):
        theta = a + math.radians(60 * k)
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def _point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    qx = x1 + t * dx
    qy = y1 + t * dy
    return math.hypot(px - qx, py - qy)


def _hexagon_min_dist(v1, v2):
    """Minimum distance between two convex hexagons (lists of 6 (x,y) tuples).
    If overlapping, return negative of max penetration (approx)."""
    # Check if any vertex of v1 is inside v2 or vice versa
    def _point_in_poly(px, py, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # If vertices are inside each other, they overlap
    for i in range(6):
        if _point_in_poly(v1[i][0], v1[i][1], v2):
            return -1.0
        if _point_in_poly(v2[i][0], v2[i][1], v1):
            return -1.0

    # Minimum distance between edges
    min_d = float('inf')
    for i in range(6):
        x1, y1 = v1[i]
        x2, y2 = v1[(i + 1) % 6]
        for j in range(6):
            x3, y3 = v2[j]
            x4, y4 = v2[(j + 1) % 6]
            # distance from point to segment for both directions
            d1 = _point_to_segment_dist(x1, y1, x3, y3, x4, y4)
            d2 = _point_to_segment_dist(x3, y3, x1, y1, x2, y2)
            min_d = min(min_d, d1, d2)
    return min_d


def _outer_hex_side_length(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute the minimum side length of a regular hexagon (at given center and angle)
    that contains all inner hexagons."""
    cx, cy = outer_center
    oa = math.radians(outer_angle)
    # The outer hexagon has 6 edges. For a regular hexagon with side s,
    # the distance from center to each edge (apothem) = s * sqrt(3)/2
    # We need to find the smallest s such that all inner vertices are inside.
    # For a regular hexagon centered at origin with angle 0 (flat-top),
    # the edges are at angles 0, 60, 120, 180, 240, 300 from center,
    # and the apothem direction for edge k is at angle 60*k + 30 degrees.
    
    max_required = 0.0
    # Collect all inner vertices
    all_verts = []
    for k in range(N):
        vx, vy, va = inner_positions[k], inner_angles[k]
        verts = _hex_vertices(vx, vy, 1.0, va)
        all_verts.extend(verts)
    
    # For each of the 6 edge directions of the outer hexagon,
    # the constraint is: for each vertex, the projection onto the outward
    # normal of that edge must be <= apothem = s * sqrt(3)/2
    # Edge normals are at angles: outer_angle + 30 + 60*k for k=0..5
    for k in range(6):
        normal_angle = oa + math.radians(30 + 60 * k)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        # Max projection of any inner vertex relative to outer center
        max_proj = max((vx - cx) * nx + (vy - cy) * ny for vx, vy in all_verts)
        # Also need the opposite direction (the edge on the other side)
        # Actually for a hexagon, each edge has one outward normal.
        # The constraint is max_proj <= s * sqrt(3)/2
        required_s = max_proj / (math.sqrt(3) / 2)
        max_required = max(max_required, required_s)
    
    # Also check the other 6 directions (inward normals give same constraint by symmetry)
    # Actually, a regular hexagon is defined by 6 half-planes. Each edge normal points outward.
    # We need all points to satisfy: dot(p - c, n_k) <= apothem for all 6 edge normals.
    # But we also need dot(p - c, -n_k) <= apothem, which is the same as the opposite edge.
    # So we just need to check all 6 outward normals.
    
    return max(0.0, max_required)


def _objective(params):
    """Objective: outer hexagon side length. params = [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10]"""
    if _elapsed() > 150.0 - DEADLINE_MARGIN:
        return 1e6
    
    inner_positions = []
    inner_angles = []
    for k in range(N):
        x = params[3*k]
        y = params[3*k+1]
        a = params[3*k+2]
        inner_positions.append((x, y))
        inner_angles.append(a)
    
    # Check pairwise distances (disjointness)
    for i in range(N):
        for j in range(i+1, N):
            xi, yi = inner_positions[i]
            xj, yj = inner_positions[j]
            ai = inner_angles[i]
            aj = inner_angles[j]
            vi = _hex_vertices(xi, yi, 1.0, ai)
            vj = _hex_vertices(xj, yj, 1.0, aj)
            d = _hexagon_min_dist(vi, vj)
            if d < 0.01:  # overlapping or too close
                return 1e6 + (0.01 - d) * 1000
    
    # Find minimum enclosing hexagon: try a few outer angles
    best_s = float('inf')
    for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
        # Optimize outer center
        def _outer_obj(c):
            s = _outer_hex_side_length(inner_positions, inner_angles, c, oa)
            return s
        # Initial guess: centroid
        cx0 = sum(p[0] for p in inner_positions) / N
        cy0 = sum(p[1] for p in inner_positions) / N
        res = minimize(_outer_obj, [cx0, cy0], method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        s = _outer_hex_side_length(inner_positions, inner_angles, (res.x[0], res.x[1]), oa)
        best_s = min(best_s, s)
    
    return best_s


def construct():
    if _elapsed() > 150.0 - DEADLINE_MARGIN:
        # Fallback: simple lattice
        inner = [[1.732 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 20.0, 0.0
    
    # Initial configuration: hexagonal lattice arrangement
    # Center at origin, 6 around it, plus 4 more
    # Hex lattice spacing for unit hexagons: distance between centers = 2 * apothem = sqrt(3)
    # For flat-top hexagons, neighbors are at:
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    sqrt3 = math.sqrt(3)
    
    # Central hexagon
    init = [0.0, 0.0, 0.0]
    # Ring of 6
    ring_angles = [0, 60, 120, 180, 240, 300]
    for a in ring_angles:
        rad = math.radians(a)
        x = sqrt3 * math.cos(rad)
        y = sqrt3 * math.sin(rad)
        init.extend([x, y, 0.0])
    # 4 more: extend outward from 4 of the ring hexagons
    for a in [0, 120, 240, 60]:
        rad = math.radians(a)
        x = 2 * sqrt3 * math.cos(rad)
        y = 2 * sqrt3 * math.sin(rad)
        init.extend([x, y, 0.0])
    
    # Add small gap to ensure disjointness
    # Scale positions slightly
    scale = 1.001
    for i in range(0, len(init), 3):
        init[i] *= scale
        init[i+1] *= scale
    
    best_params = init[:]
    best_score = _objective(best_params)
    
    # Local optimization
    if _elapsed() < 150.0 - DEADLINE_MARGIN:
        res = minimize(_objective, best_params, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    # Extract result
    inner = []
    for k in range(N):
        x = best_params[3*k]
        y = best_params[3*k+1]
        a = best_params[3*k+2]
        inner.append([x, y, a])
    
    # Find best outer center and angle
    best_outer = (0.0, 0.0, best_score, 0.0)
    inner_positions = [(inner[k][0], inner[k][1]) for k in range(N)]
    inner_angles = [inner[k][2] for k in range(N)]
    
    for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
        cx0 = sum(p[0] for p in inner_positions) / N
        cy0 = sum(p[1] for p in inner_positions) / N
        def _outer_obj(c, oa=oa):
            return _outer_hex_side_length(inner_positions, inner_angles, c, oa)
        res_o = minimize(_outer_obj, [cx0, cy0], method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
        s = _outer_hex_side_length(inner_positions, inner_angles, (res_o.x[0], res_o.x[1]), oa)
        if s < best_outer[2]:
            best_outer = (res_o.x[0], res_o.x[1], s, oa)
    
    return inner, (best_outer[0], best_outer[1]), best_outer[2], best_outer[3]
