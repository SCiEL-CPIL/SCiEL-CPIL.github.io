"""Packing unit regular hexagons inside a regular hexagon using trust-region optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle_rad
    vertices = np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])
    return vertices

def signed_distance_point_to_polygon(point, polygon_vertices):
    """
    Compute signed distance from point to polygon.
    Positive if point is outside, negative if inside.
    Uses ray casting and distance to edges.
    """
    px, py = point
    n = len(polygon_vertices)
    inside = False
    min_dist = float('inf')
    
    for i in range(n):
        x1, y1 = polygon_vertices[i]
        x2, y2 = polygon_vertices[(i + 1) % n]
        
        # Ray casting for inside/outside
        if ((y1 > py) != (y2 > py)):
            x_intersect = (px - x1) * (y2 - y1) / (y2 - y1) + x1
            if px < x_intersect:
                inside = not inside
        
        # Distance to edge
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq > 1e-12:
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
            min_dist = min(min_dist, dist)
    
    if min_dist == float('inf'):
        return 0.0
    
    return min_dist if inside else -min_dist

def polygon_overlap_distance(poly1, poly2):
    """
    Estimate the minimum distance needed to separate two convex polygons.
    Returns 0 if they overlap, positive if separated.
    Uses support function approximation.
    """
    # Check if any vertex of poly1 is inside poly2 or vice versa
    for v in poly1:
        if signed_distance_point_to_polygon(v, poly2) < -1e-6:
            return signed_distance_point_to_polygon(v, poly2)  # negative means overlap
    
    for v in poly2:
        if signed_distance_point_to_polygon(v, poly1) < -1e-6:
            return signed_distance_point_to_polygon(v, poly1)  # negative means overlap
    
    # If no vertex inside, compute minimum distance between edges
    min_dist = float('inf')
    for i in range(len(poly1)):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % len(poly1)]
        for j in range(len(poly2)):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % len(poly2)]
            d = _segment_distance(p1, p2, p3, p4)
            min_dist = min(min_dist, d)
    
    return min_dist

def _segment_distance(p1, p2, p3, p4):
    """Compute minimum distance between two line segments."""
    # Simple approach: sample points along both segments
    n_samples = 10
    min_d = float('inf')
    for i in range(n_samples + 1):
        t = i / n_samples
        x1 = p1[0] + t * (p2[0] - p1[0])
        y1 = p1[1] + t * (p2[1] - p1[1])
        for j in range(n_samples + 1):
            s = j / n_samples
            x2 = p3[0] + s * (p4[0] - p3[0])
            y2 = p3[1] + s * (p4[1] - p3[1])
            d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            min_d = min(min_d, d)
    return min_d

def construct():
    n = N
    # Initial guess: hexagonal lattice arrangement
    # For n hexagons, arrange in rows
    # Row lengths: for n=11, we can do 3+4+4 or 4+3+4 etc.
    
    # Let's use a hexagonal packing pattern
    # Center at origin, place hexagons in a hexagonal grid
    # Distance between centers of touching unit hexagons = 2 * sqrt(3) * (1/2) * 2 = 2*sqrt(3)? 
    # Actually, for regular hexagons of side 1, the distance from center to vertex is 1.
    # Distance between centers of two touching hexagons = 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let's place them in a hexagonal pattern
    positions = []
    # For n=11: 1 center, 6 around, 4 more
    # Or just use a linear arrangement initially and let optimizer improve
    
    # Simple initial: spread out in a grid-like pattern
    dx = 2 * np.sqrt(3)  # distance between centers in hex packing
    dy = 3.0  # vertical spacing for hex packing
    
    # Place in rows
    row_lengths = [3, 5, 3] if n == 11 else [n]
    idx = 0
    for r, row_len in enumerate(row_lengths):
        row_offset = (len(row_lengths) - 1 - r) * dy
        for c in range(row_len):
            x = c * dx - (row_len - 1) * dx / 2
            y = row_offset - (len(row_lengths) - 1) * dy / 2
            positions.append([x, y, 0.0])
            idx += 1
    
    # If we have more than n, truncate; if less, add
    while len(positions) < n:
        positions.append([0.0, 0.0, 0.0])
    positions = positions[:n]
    
    # Variables: for each hexagon, (x, y, angle_deg)
    # Plus outer hexagon: (cx, cy, side, angle_deg)
    # Total variables: 3*n + 4
    
    def unpack(params):
        inner_params = params[:3*n]
        outer_params = params[3*n:]
        inner = []
        for i in range(n):
            x = inner_params[3*i]
            y = inner_params[3*i + 1]
            angle = inner_params[3*i + 2]
            inner.append([x, y, angle])
        cx = outer_params[0]
        cy = outer_params[1]
        side = outer_params[2]
        outer_angle = outer_params[3]
        return inner, (cx, cy), side, outer_angle
    
    def objective(params):
        inner, (cx, cy), side, outer_angle = unpack(params)
        
        if side <= 0:
            return 1e6
        
        # Compute outer hexagon vertices
        outer_poly = hexagon_vertices(cx, cy, side, outer_angle)
        
        # For each inner hexagon, compute its vertices
        inner_polys = []
        for i, (x, y, angle) in enumerate(inner):
            poly = hexagon_vertices(x, y, 1.0, angle)
            inner_polys.append(poly)
        
        # Objective: minimize side length
        # Constraints: 
        # 1. Each inner hexagon must be inside outer hexagon
        # 2. Inner hexagons must not overlap
        
        # We'll use a penalty-based approach
        penalty = 0.0
        
        # Check each inner hexagon is inside outer
        for poly in inner_polys:
            for v in poly:
                sd = signed_distance_point_to_polygon(v, outer_poly)
                if sd < 0:  # inside is negative, so we want sd <= 0 for all vertices
                    # Actually we want all vertices inside, so sd should be <= 0
                    # If sd > 0, vertex is outside
                    pass
                if sd > 0:
                    penalty += sd * sd
        
        # Check pairwise non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                d = polygon_overlap_distance(inner_polys[i], inner_polys[j])
                if d < 0:  # overlap
                    penalty += d * d
        
        # Return side length + penalty
        return side + 1000.0 * penalty
    
    # Initial parameters
    params = np.zeros(3 * n + 4)
    for i, (x, y, angle) in enumerate(positions):
        params[3*i] = x
        params[3*i + 1] = y
        params[3*i + 2] = angle
    params[3*n] = 0.0  # cx
    params[3*n + 1] = 0.0  # cy
    params[3*n + 2] = 10.0  # initial side estimate
    params[3*n + 3] = 0.0  # outer angle
    
    # Use Nelder-Mead for robustness with non-smooth objective
    result = minimize(objective, params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
    
    best_params = result.x
    inner, (cx, cy), side, outer_angle = unpack(best_params)
    
    # Ensure side is positive
    if side <= 0:
        side = 1.0
    
    return inner, (cx, cy), side, outer_angle

# EVOLVE-BLOCK-END
