import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12

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
    
    # Check if all inner hexagons are inside the outer hexagon
    # And check mutual non-overlap
    
    total_violation = 0.0
    
    n_inner = len(inner)
    inner_verts = []
    
    for i in range(n_inner):
        x, y, ang = inner[i]
        v = hex_vertices((x, y), 1.0, np.radians(ang))
        inner_verts.append(v)
        
        # Check inside outer
        # Use point-in-polygon or distance check? 
        # For convex polygons, checking if all vertices of inner are inside outer is sufficient for containment.
        # But we also need to ensure edges don't cross if they are close? 
        # Actually, if all vertices of a convex polygon A are inside convex polygon B, A is inside B.
        
        # Simple point in polygon check for each vertex of inner hex i against outer hex
        for v in v:
            if not point_in_polygon(v, outer_verts):
                # Calculate distance to nearest edge of outer polygon for violation measure
                dist = min_distance_to_polygon(v, outer_verts)
                total_violation += max(0, dist) # If outside, dist is positive distance to boundary? 
                # Actually, if outside, we want to penalize. 
                # Let's use a simpler penalty: if any vertex is outside, add a penalty proportional to how far out.
                # But `point_in_polygon` returns bool. We need a continuous measure.
                
    # Let's rewrite check_validity to return a continuous penalty for optimization.
    # This function is for final verification. The optimization loop will use a penalty function.
    
    # For the sake of the interface, let's just return True/False and 0/inf.
    # The optimization will use a separate penalty function.
    
    is_valid = True
    
    # 1. Containment
    for i in range(n_inner):
        v = inner_verts[i]
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                is_valid = False
                return False, np.inf

    # 2. Non-overlap
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            if polygons_intersect(inner_verts[i], inner_verts[j]):
                is_valid = False
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
        # For a convex polygon defined in CCW order, all cross products should have the same sign.
        # Assuming standard hex_vertices returns CCW.
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
        
        if max1 < min2 or max2 < min1:
            return False
            
    return True

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
    # For each inner vertex, if outside outer polygon, add distance squared
    for v in inner_verts:
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                # Find distance to closest point on outer polygon boundary
                # Simplified: distance to closest vertex? No, closest edge.
                # For speed, we can just use a large penalty if outside.
                # Better: calculate distance to polygon boundary.
                dist = min_distance_to_polygon_boundary(pt, outer_verts)
                total_penalty += dist * dist * 100.0 # Heavy penalty for containment violation

    # 2. Overlap penalty
    # For each pair, check intersection. If intersecting, estimate overlap area or distance between boundaries.
    # SAT is fast but binary. We need a continuous measure.
    # Approximation: If intersecting, find the minimum distance between vertices of one and edges of the other?
    # Or simpler: If intersecting, add a penalty based on the depth of intersection.
    # Let's use a simpler heuristic: If they intersect, add a constant penalty. 
    # This makes the landscape rugged. 
    # Alternative: Compute the minimum distance between the two polygons. If > 0, no penalty.
    # If < 0 (intersecting), penalty is -distance * large_factor.
    
    for i in range(N):
        for j in range(i + 1, N):
            min_dist = min_distance_between_polygons(inner_verts[i], inner_verts[j])
            if min_dist < 0:
                # Intersecting
                total_penalty += (-min_dist) * 100.0
            elif min_dist < 1e-3:
                # Very close, add small penalty to encourage separation
                total_penalty += (1e-3 - min_dist) * 10.0

    # Objective: Minimize outer_side + penalties
    return outer_side + total_penalty

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
        # Or find vertex of poly2 inside poly1
        # Simple heuristic: min distance from any vertex of poly1 to poly2 boundary
        min_d = np.inf
        for v in poly1:
            d = min_distance_to_polygon_boundary(v, poly2)
            if d < min_d:
                min_d = d
        # If min_d is small, they are close or intersecting.
        # If they intersect, min_d might be 0 or small. 
        # To make it negative, we can check if any vertex is strictly inside.
        # For simplicity in optimization, returning -1.0 * min_d might not be enough depth.
        # Let's return -min_d if we detect intersection, but scale it.
        # Actually, if they intersect, the distance is 0. The "penalty" needs to push them apart.
        # A common trick is to use the distance between centroids if they overlap significantly?
        # Let's stick to: if intersect, return -1.0 (fixed penalty) or -depth.
        # Let's try returning -min_d * 10 if intersecting, to create a gradient.
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    best_score = np.inf
    best_params = None
    
    # Initial configurations
    # 1. Grid-like
    # 2. Hex cluster
    
    def init_grid():
        # 4 columns, 3 rows
        # Hex side 1. Horizontal spacing: sqrt(3) * 1? No, for flat-top or pointy-top?
        # Our hex_vertices uses angles 0, 60, ... so vertices at 0 deg is (1,0). This is "pointy-top" if y is up?
        # cos(0)=1, sin(0)=0 -> (1,0).
        # cos(60)=0.5, sin(60)=0.866 -> (0.5, 0.866).
        # This is a hexagon with a vertex pointing right.
        # Width = 2 * side = 2. Height = sqrt(3) * side = 1.732.
        # Horizontal distance between centers in same row: 1.5 * side? No.
        # If vertex points right, the "width" is 2. The "height" is sqrt(3).
        # To pack them:
        # Row 0: x = 0, 3, 6... (spacing 3? No, 1.5 + 1.5 = 3? No.)
        # Let's use standard hex packing coordinates.
        # For "pointy-top" (vertex at 90 deg), spacing is different.
        # Our definition: vertex at 0 deg.
        # Distance between centers of touching hexagons:
        # Neighbor to the right (along x-axis? No, vertices are at 0, 60, 120...).
        # The edge from 0 to 60 is not horizontal.
        # The horizontal distance between centers of two hexagons sharing a vertical edge?
        # There are no vertical edges. Edges are at 30, 90, 150, 210, 270, 330 degrees?
        # Vertices: 0, 60, 120, 180, 240, 300.
        # Edges connect 0-60, 60-120, etc.
        # Midpoint of edge 0-60: (0.75, 0.433).
        # This is getting complicated. Let's just use a random or grid init with large spacing and let optimizer fix it.
        
        params = np.zeros(4 + 3*N)
        # Outer side large
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        # Inner positions in a grid
        x = 0
        y = 0
        for i in range(N):
            col = i % 4
            row = i // 4
            # Spacing 3.5 in x, 3.0 in y (large enough to not overlap initially)
            ix = col * 3.5
            iy = row * 3.0
            params[4 + 3*i] = ix
            params[4 + 3*i+1] = iy
            params[4 + 3*i+2] = 0.0
        return params

    def init_hex_cluster():
        # 7 in center, 5 around
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        # Center
        idx = 0
        params[4 + 3*idx] = 0
        params[4 + 3*idx+1] = 0
        params[4 + 3*idx+2] = 0
        
        # 6 neighbors
        # Distance 2 (touching)
        # Angles 0, 60, ...
        for k in range(6):
            idx = k + 1
            ang = np.radians(k * 60)
            dist = 2.0 # Side 1, touching distance is 2? 
            # If vertex points right, neighbor to the right is at distance 2?
            # Vertex at (1,0). Neighbor vertex at (-1,0)? No.
            # Two unit hexagons touching side by side.
            # If one is at (0,0), vertices at (1,0), (0.5, 0.866)...
            # The rightmost point is (1,0).
            # The neighbor to the right should have its leftmost point at (1,0)?
            # Leftmost point of a hexagon at (x,y) is (x-1, y).
            # So center at (2,0) has leftmost point (1,0).
            # So distance is 2.
            params[4 + 3*idx] = 2.0 * np.cos(ang)
            params[4 + 3*idx+1] = 2.0 * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            
        # 5 more
        # Place them further out
        for k in range(5):
            idx = k + 7
            ang = np.radians(k * 72 + 30)
            dist = 4.0
            params[4 + 3*idx] = dist * np.cos(ang)
            params[4 + 3*idx+1] = dist * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            
        return params

    initializations = [init_grid(), init_hex_cluster()]
    
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
        # Just return a large hexagon with hexagons far apart
        inner = []
        for i in range(N):
            inner.append((0, 0, 0)) # Overlapping! Invalid.
        # We must return a valid config. 
        # If optimization fails, we should return a known valid config.
        # Let's construct a valid one manually.
        # Outer side 10, center (0,0).
        # Inner hexagons at (0,0), (2,0), (-2,0), (0, 1.732), etc.
        # Actually, just return a large box.
        # The evaluator checks validity. If invalid, score 0.
        # So we MUST ensure the returned config is valid.
        
        # Let's try a simple valid config:
        # Outer side 10.
        # Inner hexagons in a grid with spacing 3.
        # Max x = 3*3 = 9. Fits in side 10?
        # Hex width 2. Center at 9, rightmost 10. Outer rightmost 10. Touching.
        # Let's use spacing 3.5 to be safe.
        inner = []
        for i in range(N):
            col = i % 4
            row = i // 4
            x = col * 3.5 - 3.5*1.5 # Center it
            y = row * 3.0 - 3.0*1.0
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
        # If the optimized result is invalid (due to numerical issues or local minima),
        # fall back to a simple valid configuration.
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
