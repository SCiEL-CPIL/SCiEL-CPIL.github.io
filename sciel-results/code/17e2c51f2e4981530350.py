import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12
SQRT3 = np.sqrt(3.0)
APOTHEM = SQRT3 / 2.0

def hex_vertices(center, side, angle):
    """
    Calculate the vertices of a regular hexagon.
    center: (x, y)
    side: side length
    angle: rotation in radians
    """
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + angle
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def check_validity(inner, outer_center, outer_side, outer_angle):
    """
    Check if inner hexagons are disjoint and inside the outer hexagon.
    inner: list of (x, y, angle_degrees)
    outer_center: (x, y)
    outer_side: float
    outer_angle: float (degrees)
    
    Returns:
        is_valid (bool)
        violation (float) - 0 if valid, >0 if invalid (measure of violation)
    """
    if outer_side <= 0:
        return False, np.inf

    outer_verts = hex_vertices(outer_center, outer_side, np.radians(outer_angle))
    
    n_inner = len(inner)
    inner_verts = []
    
    for i in range(n_inner):
        x, y, ang = inner[i]
        v = hex_vertices((x, y), 1.0, np.radians(ang))
        inner_verts.append(v)
        
    # 1. Containment
    for i in range(n_inner):
        v = inner_verts[i]
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                return False, np.inf

    # 2. Non-overlap
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            if polygons_intersect(inner_verts[i], inner_verts[j]):
                return False, np.inf
                
    return True, 0.0

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a convex polygon.
    polygon: Nx2 array of vertices.
    """
    x, y = point
    n = len(polygon)
    inside = True
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Cross product to check if point is on the same side of all edges
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross < -1e-9: # Tolerance
            return False
    return True

def polygons_intersect(poly1, poly2):
    """
    Check if two convex polygons intersect.
    Uses Separating Axis Theorem (SAT) simplified for convex polygons.
    """
    def project_polygon(axis, poly):
        projections = np.dot(poly, axis)
        return np.min(projections), np.max(projections)

    def get_edges(poly):
        edges = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            # Normal vector
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal /= norm
            edges.append(normal)
        return edges

    axes = get_edges(poly1) + get_edges(poly2)
    
    for axis in axes:
        min1, max1 = project_polygon(axis, poly1)
        min2, max2 = project_polygon(axis, poly2)
        
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
            
    return True

def min_distance_to_polygon_boundary(point, polygon):
    """
    Calculate minimum distance from a point to the boundary of a convex polygon.
    """
    min_d = np.inf
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        d = distance_point_to_segment(point, p1, p2)
        if d < min_d:
            min_d = d
    return min_d

def distance_point_to_segment(p, a, b):
    """
    Distance from point p to segment ab.
    """
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-9:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_len_sq
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def min_distance_between_polygons(poly1, poly2):
    """
    Approximate minimum distance between two convex polygons.
    If they intersect, return a negative value (depth of intersection).
    This is a heuristic for optimization continuity.
    """
    # First check intersection
    if polygons_intersect(poly1, poly2):
        # Estimate depth: find vertex of poly1 inside poly2 and measure distance to boundary
        min_d = np.inf
        for v in poly1:
            d = min_distance_to_polygon_boundary(v, poly2)
            if d < min_d:
                min_d = d
        for v in poly2:
            d = min_distance_to_polygon_boundary(v, poly1)
            if d < min_d:
                min_d = d
        return -min_d * 10.0
    
    # If not intersecting, find min distance between any vertex-edge pair
    min_d = np.inf
    for v in poly1:
        d = min_distance_to_polygon_boundary(v, poly2)
        if d < min_d:
            min_d = d
    for v in poly2:
        d = min_distance_to_polygon_boundary(v, poly1)
        if d < min_d:
            min_d = d
    return min_d

def penalty_function(params):
    """
    Objective function for optimization.
    params: 
        - outer_side (1)
        - outer_center (2)
        - outer_angle (1)
        - inner centers (2*N)
        - inner angles (N)
    """
    outer_side = params[0]
    outer_cx, outer_cy = params[1:3]
    outer_angle = params[3]
    
    inner_params = params[4:]
    inner = []
    for i in range(N):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        a = inner_params[3*i+2]
        inner.append((x, y, a))
        
    outer_center = (outer_cx, outer_cy)
    
    # Compute vertices for all inner hexagons
    inner_verts = []
    for i in range(N):
        x, y, a = inner[i]
        v = hex_vertices((x, y), 1.0, np.radians(a))
        inner_verts.append(v)
        
    outer_verts = hex_vertices(outer_center, outer_side, np.radians(outer_angle))
    
    total_penalty = 0.0
    
    # 1. Containment penalty
    for v in inner_verts:
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                dist = min_distance_to_polygon_boundary(pt, outer_verts)
                total_penalty += dist * dist * 100.0

    # 2. Overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            min_dist = min_distance_between_polygons(inner_verts[i], inner_verts[j])
            if min_dist < 0:
                total_penalty += (-min_dist) * 100.0
            elif min_dist < 1e-3:
                total_penalty += (1e-3 - min_dist) * 10.0

    # Objective: Minimize outer_side + penalties
    return outer_side + total_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    best_score = np.inf
    best_params = None
    
    # Initial configurations
    # 1. Grid-like
    # 2. Hex cluster
    # 3. Honeycomb (from inspiration)
    
    def init_grid():
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        x = 0
        y = 0
        for i in range(N):
            col = i % 4
            row = i // 4
            ix = col * 3.5
            iy = row * 3.0
            params[4 + 3*i] = ix
            params[4 + 3*i+1] = iy
            params[4 + 3*i+2] = 0.0
        return params

    def init_hex_cluster():
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        idx = 0
        params[4 + 3*idx] = 0
        params[4 + 3*idx+1] = 0
        params[4 + 3*idx+2] = 0
        
        for k in range(6):
            idx = k + 1
            ang = np.radians(k * 60)
            dist = 2.0
            params[4 + 3*idx] = dist * np.cos(ang)
            params[4 + 3*idx+1] = dist * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            
        for k in range(5):
            idx = k + 7
            ang = np.radians(k * 72 + 30)
            dist = 4.0
            params[4 + 3*idx] = dist * np.cos(ang)
            params[4 + 3*idx+1] = dist * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            
        return params

    def init_honeycomb():
        # 3x4 staggered grid, tighter spacing
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        for i in range(N):
            row = i // 4
            col = i % 4
            # Spacing: sqrt(3) ~ 1.732 horizontally, 1.5 vertically
            # Staggered: odd rows shift by sqrt(3)/2
            x = col * SQRT3 * 1.05
            y = row * 1.55
            if row % 2 == 1:
                x += SQRT3 * 1.05 / 2
            params[4 + 3*i] = x
            params[4 + 3*i+1] = y
            params[4 + 3*i+2] = 0.0
        return params

    initializations = [init_grid(), init_hex_cluster(), init_honeycomb()]
    
    for init_params in initializations:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(penalty_function, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        except Exception:
            pass
            
    if best_params is None:
        # Fallback: return a valid but large configuration
        inner = []
        for i in range(N):
            col = i % 4
            row = i // 4
            x = col * 3.5 - 5.0
            y = row * 3.0 - 1.5
            inner.append((x, y, 0))
        return inner, (0, 0), 10.0, 0.0

    # Extract best params
    outer_side = best_params[0]
    outer_cx = best_params[1]
    outer_cy = best_params[2]
    outer_angle = best_params[3]
    
    inner = []
    for i in range(N):
        x = best_params[4 + 3*i]
        y = best_params[4 + 3*i+1]
        a = best_params[4 + 3*i+2]
        inner.append((x, y, a))
        
    outer_center = (outer_cx, outer_cy)
    
    # Final verification
    valid, _ = check_validity(inner, outer_center, outer_side, outer_angle)
    
    if not valid:
        # If the optimized result is invalid, fall back to a simple valid configuration.
        # Use the honeycomb init with a sufficiently large outer hexagon.
        # Calculate required outer side for the honeycomb config.
        # Max extent in x: 3 * SQRT3 * 1.05 + SQRT3/2*1.05 + 1 (radius)
        # Max extent in y: 2 * 1.55 + 1
        # Let's just use a safe large side.
        inner = []
        for i in range(N):
            row = i // 4
            col = i % 4
            x = col * SQRT3 * 1.05
            y = row * 1.55
            if row % 2 == 1:
                x += SQRT3 * 1.05 / 2
            # Center the cluster
            x -= (3 * SQRT3 * 1.05 + SQRT3 * 1.05 / 2) / 2
            y -= (2 * 1.55) / 2
            inner.append((x, y, 0))
        outer_side = 6.0
        outer_center = (0, 0)
        outer_angle = 0.0
        
        # Verify this fallback
        valid, _ = check_validity(inner, outer_center, outer_side, outer_angle)
        if not valid:
            # Last resort: very large box
            inner = []
            for i in range(N):
                col = i % 4
                row = i // 4
                x = col * 3.5 - 5.0
                y = row * 3.0 - 1.5
                inner.append((x, y, 0))
            outer_side = 10.0
            outer_center = (0, 0)
            outer_angle = 0.0

    return inner, outer_center, outer_side, outer_angle
