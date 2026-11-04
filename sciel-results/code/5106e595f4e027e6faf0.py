import math
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 11
SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap.
    """
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _min_enclosing_hex_side(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon, given a fixed outer angle.
    """
    n = len(inner_params)
    apothem_factor = SQRT3 / 2.0
    
    # Edge normals for outer hexagon with rotation outer_angle_deg
    # Normals are at angle + 30, + 90, etc?
    # Vertices are at angle. Normals are at angle + 30.
    normal_angles = np.deg2rad(outer_angle_deg + 30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_val = 0.0
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            projections = normals @ v
            for proj in projections:
                if proj > 0:
                    required_s = proj / apothem_factor
                    if required_s > max_val:
                        max_val = required_s
    
    return max_val

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if configuration is valid."""
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < min_sep:
                return False
    
    return True

def _generate_lattice_cluster():
    """
    Generate the 1-6-4 lattice cluster.
    1 center, 6 neighbors, 4 from the second ring (distance 3).
    """
    v1 = np.array([SQRT3, 0.0])
    v2 = np.array([SQRT3/2, 1.5])
    
    # Center
    centers = [np.array([0.0, 0.0])]
    
    # 6 neighbors
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    for i, j in neighbors:
        centers.append(i * v1 + j * v2)
        
    # 4 from second ring (distance 3)
    # The 6 points at distance 3 are: (1,1), (2,-1), (1,-2), (-1,-1), (-2,1), (-1,2)
    # We pick 4 of them. To maintain symmetry, we pick a subset.
    # Let's pick (1,1), (-1,-1), (1,-2), (-1,2) -> Angles 30, 210, -30(330), 150.
    # This leaves gaps at 90 and 270.
    # Alternatively, pick (1,1), (2,-1), (-1,-1), (-2,1).
    # Let's just pick 4 arbitrary ones and let the optimizer handle it.
    # Actually, for the 1D rotation optimization, the specific choice matters.
    # The most symmetric choice for 4 out of 6 is to remove 2 opposite ones?
    # If we remove 2 opposite, we have 4 left, which form a rectangle?
    # Let's pick indices 0, 1, 3, 4 from the list of 6.
    second_ring_candidates = [
        (1, 1), (2, -1), (1, -2), (-1, -1), (-2, 1), (-1, 2)
    ]
    # Pick 0, 2, 3, 5?
    selected_second = [second_ring_candidates[0], second_ring_candidates[2], 
                       second_ring_candidates[3], second_ring_candidates[5]]
    
    for i, j in selected_second:
        centers.append(i * v1 + j * v2)

    inner = []
    for c in centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    return np.array(inner)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    # Generate the lattice cluster
    inner_params = _generate_lattice_cluster()
    
    # Optimize the rotation angle of the container
    # The side length is a function of the angle.
    # We minimize this function.
    
    def side_length_for_angle(angle_deg):
        return _min_enclosing_hex_side(inner_params, angle_deg)
    
    # Initial coarse search
    best_angle = 0.0
    best_side = float('inf')
    
    # The function is periodic with period 60.
    # We search in [0, 60).
    for angle in np.arange(0, 60, 1.0):
        if time.time() > DEADLINE - safety_margin:
            break
        s = side_length_for_angle(angle)
        if s < best_side:
            best_side = s
            best_angle = angle
            
    # Refine with minimize_scalar
    if time.time() < DEADLINE - safety_margin:
        try:
            res = minimize_scalar(side_length_for_angle, 
                                  bounds=(best_angle - 5, best_angle + 5), 
                                  method='bounded', 
                                  options={'xatol': 1e-10})
            if res.success:
                best_side = res.fun
                best_angle = res.x
        except Exception:
            pass
            
    # Final validation
    final_side = _min_enclosing_hex_side(inner_params, best_angle)
    if not _is_valid(inner_params, final_side, best_angle, min_sep=1e-8):
        # Fallback: expand side slightly
        final_side *= 1.001
        
    inner_rows = [[float(row[0]), float(row[1]), float(row[2])] for row in inner_params]
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(best_angle)
    
    return inner_rows, center, side, angle_deg
