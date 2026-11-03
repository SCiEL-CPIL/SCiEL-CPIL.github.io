import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _point_in_convex_polygon(pt, poly):
    """Check if point pt is inside convex polygon poly (list of vertices)."""
    sign = 0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        cp = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
        if cp > 1e-9:
            if sign == -1:
                return False
            sign = 1
        elif cp < -1e-9:
            if sign == 1:
                return False
            sign = -1
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def orient(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val > 1e-9: return 1
        if val < -1e-9: return -1
        return 0

    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    return False

def _min_dist_point_to_segment(pt, p1, p2):
    """Minimum distance from point pt to segment p1-p2."""
    px, py = pt
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)

def _penalty_overlap(v1, v2):
    """
    Calculate penalty for overlap between two hexagons defined by vertices v1, v2.
    Returns 0 if disjoint, positive value if overlapping.
    """
    min_dist = float('inf')
    for pt in v1:
        if _point_in_convex_polygon(pt, v2):
            for i in range(6):
                d = _min_dist_point_to_segment(pt, v2[i], v2[(i+1)%6])
                if d < min_dist:
                    min_dist = d
    for pt in v2:
        if _point_in_convex_polygon(pt, v1):
            for i in range(6):
                d = _min_dist_point_to_segment(pt, v1[i], v1[(i+1)%6])
                if d < min_dist:
                    min_dist = d
    
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return 0.1 

    if min_dist < float('inf'):
        return min_dist
    return 0.0

def _objective(params):
    """
    params: [x1, y1, a1, ..., x11, y11, a11, outer_angle]
    Returns side length + penalty.
    """
    if time.time() > _DEADLINE - 2.0:
        return 1e9

    inner_hexes = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3 + 1]
        a = params[i*3 + 2]
        inner_hexes.append((x, y, a))
    
    outer_angle = params[N*3]

    all_verts = []
    hex_verts_list = []
    for x, y, a in inner_hexes:
        verts = _get_hex_vertices(x, y, a)
        all_verts.extend(verts)
        hex_verts_list.append(verts)

    r = 0.0
    for k in range(6):
        a = math.radians(outer_angle + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_verts:
            proj = vx * nx + vy * ny
            if abs(proj) > r:
                r = abs(proj)
    
    side = r / (math.sqrt(3) / 2)

    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            p = _penalty_overlap(hex_verts_list[i], hex_verts_list[j])
            penalty += p * p

    return side + 1000.0 * penalty

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    sqrt3 = math.sqrt(3)
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * sqrt3 + j * sqrt3 / 2
            y = j * 1.5
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    initial_params = []
    for idx in range(N):
        x, y = candidates[idx][1], candidates[idx][2]
        initial_params.extend([x, y, 0.0])
    initial_params.append(0.0)

    best_result = None
    best_score = float('inf')

    # Try a few perturbations
    for restart in range(3):
        if time.time() > _DEADLINE - 20.0:
            break
        
        params = list(initial_params)
        if restart > 0:
            # Perturb
            for i in range(len(params)):
                params[i] += (np.random.random() - 0.5) * 0.1

        try:
            res = minimize(_objective, params, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 500, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
        except Exception:
            pass

    if best_result is None:
        best_result = initial_params

    inner = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3 + 1]
        a = best_result[i*3 + 2]
        inner.append([x, y, a])
    
    outer_angle = best_result[N*3]
    
    all_verts = []
    for x, y, a in inner:
        verts = _get_hex_vertices(x, y, a)
        all_verts.extend(verts)
        
    r = 0.0
    for k in range(6):
        a = math.radians(outer_angle + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_verts:
            proj = vx * nx + vy * ny
            if abs(proj) > r:
                r = abs(proj)
    
    side = r / (math.sqrt(3) / 2)

    return inner, (0.0, 0.0), side, outer_angle

# EVOLVE-BLOCK-END
