"""Packing unit regular hexagons inside a regular hexagon using hexagonal lattice."""
import math
import numpy as np

# EVOLVE-BLOCK-START

def hex_to_cartesian(q, r, s=0.0):
    """Convert axial coordinates (q, r) to Cartesian (x, y) for a regular hex grid.
    Using pointy-top orientation where neighbors are at 60 degree angles.
    Actually, let's use flat-top orientation for consistency with standard hex grids.
    
    For flat-top hexagons with side length 1:
    - width = 2 * side = 2
    - height = sqrt(3) * side = sqrt(3)
    - x = 1.5 * side * q
    - y = side * sqrt(3) * (r + q/2)
    """
    side = 1.0
    x = 1.5 * side * q
    y = side * math.sqrt(3) * (r + q / 2.0)
    return x, y

def cartesian_to_hex(x, y):
    """Convert Cartesian to axial coordinates (approximate)."""
    side = 1.0
    q = (2.0 / 3.0) * x / side
    r = (-1.0 / 3.0) * x / side + (1.0 / math.sqrt(3)) * y / side
    return q, r

def hex_distance(q1, r1, q2, r2):
    """Compute hex distance between two axial coordinates."""
    dq = q1 - q2
    dr = r1 - r2
    ds = -dq - dr
    return max(abs(dq), abs(dr), abs(ds))

def get_hexagon_vertices(x, y, angle_degrees, side=1.0):
    """Get the 6 vertices of a regular hexagon centered at (x,y) with given rotation."""
    angle_rad = math.radians(angle_degrees)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = x + side * math.cos(theta)
        vy = y + side * math.sin(theta)
        vertices.append((vx, vy))
    return vertices

def point_in_hexagon(px, py, hx, hy, angle_degrees, side=1.0):
    """Check if point (px, py) is inside hexagon centered at (hx, hy) with rotation."""
    # Transform to hexagon's local coordinate system
    dx = px - hx
    dy = py - hy
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    
    # For a regular hexagon with side length 'side', the apothem is side * sqrt(3)/2
    # In local coords, the hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon boundary in local coords: |lx| <= side, |ly| <= side*sqrt(3)/2, 
    # and |lx| + |ly|/sqrt(3) <= side (for the hexagonal shape)
    # More precisely, the hexagon is the intersection of 6 half-planes.
    
    # A regular hexagon with side s centered at origin, rotated by 0:
    # vertices at (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The constraints are:
    # |lx| <= s
    # |ly| <= s*sqrt(3)/2
    # |lx|/s + |ly|/(s*sqrt(3)/2) <= 1 ... no that's not right
    
    # Let me use the standard: for a hexagon with "flat" top/bottom (vertices at 0, 60, ...),
    # the six sides are at angles 30, 90, 150, 210, 270, 330 from center.
    # The apothem (distance from center to each side) is s * sqrt(3) / 2.
    # The normal directions to the sides are at angles 30, 90, 150, 210, 270, 330.
    
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        theta = math.pi / 6 + i * math.pi / 3  # 30, 90, 150, 210, 270, 330 degrees
        nx = math.cos(theta)
        ny = math.sin(theta)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects with line segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # Check collinear cases (touching)
    if d1 == 0 and on_segment(p3, p4, p1):
        return True
    if d2 == 0 and on_segment(p3, p4, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, p3):
        return True
    if d4 == 0 and on_segment(p1, p2, p4):
        return True
    
    return False

def on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and \
       min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False

def hexagons_overlap(v1, v2, epsilon=1e-9):
    """Check if two hexagons (as lists of 6 vertices) overlap.
    First check if any vertex of one is inside the other, then check edge intersections.
    """
    # Check if any vertex of hex1 is inside hex2 or vice versa
    for vertex in v1:
        if point_in_polygon(vertex, v2):
            return True
    for vertex in v2:
        if point_in_polygon(vertex, v1):
            return True
    
    # Check if any edges intersect
    for i in range(6):
        for j in range(6):
            e1 = (v1[i], v1[(i+1) % 6])
            e2 = (v2[j], v2[(j+1) % 6])
            if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True
    
    return False

def point_in_polygon(point, polygon):
    """Ray casting algorithm to check if point is inside polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def compute_min_outer_hexagon(vertices, angle_degrees):
    """Given a set of points (vertices of inner hexagons), find the minimum side length
    of a regular hexagon with the given rotation that contains all points.
    
    We'll use a binary search on the side length, and for each candidate, check if
    all points fit inside a hexagon of that size centered at the optimal center.
    
    Actually, for a given rotation, the minimal enclosing hexagon can be found by
    considering that the hexagon is the intersection of 6 half-planes. The optimal
    center minimizes the maximum distance to the 6 sides.
    """
    # For a hexagon with rotation angle_degrees, the 6 side normals are at:
    # angle_degrees + 30 + k*60 for k=0..5
    # The constraint is: for each normal direction n_k, the projection of all points
    # onto n_k must be within [-apothem, +apothem] relative to the center.
    # So the center's projection onto n_k must be within [max_proj_k - apothem, min_proj_k + apothem]
    # Wait, that's not quite right. Let me think again.
    
    # For a hexagon centered at c with apothem a, a point p is inside iff:
    # |dot(p - c, n_k)| <= a for all k=0..5
    # This means: a - dot(p, n_k) <= dot(c, n_k) <= a + dot(p, n_k) for all p
    # So: max_p (dot(p, n_k) - a) <= dot(c, n_k) <= min_p (dot(p, n_k) + a) for all k
    
    # For feasibility, we need max_p(dot(p,n_k)) - min_p(dot(p,n_k)) <= 2*a for all k
    # The minimum a is therefore max over k of (max_p(dot(p,n_k)) - min_p(dot(p,n_k))) / 2
    
    # The side length s is related to apothem a by: a = s * sqrt(3) / 2
    # So s = 2 * a / sqrt(3)
    
    points = np.array(vertices)
    angle_rad = math.radians(angle_degrees)
    
    max_width = 0.0
    for k in range(6):
        theta = angle_rad + math.pi / 6 + k * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        projections = points[:, 0] * nx + points[:, 1] * ny
        width = np.max(projections) - np.min(projections)
        if width > max_width:
            max_width = width
    
    apothem = max_width / 2.0
    side_length = 2.0 * apothem / math.sqrt(3)
    
    return side_length

def find_min_outer_hexagon(points):
    """Find the minimum side length of a regular hexagon containing all given points.
    We try multiple rotations to find the best.
    """
    best_side = float('inf')
    # Try several rotation angles
    for angle_deg in range(0, 180, 5):
        side = compute_min_outer_hexagon(points, angle_deg)
        if side < best_side:
            best_side = side
    # Refine around the best
    # For simplicity, just use the coarse search result
    return best_side

def construct():
    """Construct a packing of 11 unit hexagons inside a minimal outer hexagon.
    
    Strategy: Place hexagons on a hexagonal lattice. A hexagonal number arrangement
    for N hexagons: for N=11, we can use a "flower" pattern or a linear arrangement.
    
    The hexagonal numbers are 1, 7, 19, 37, ... So 11 is not a perfect hex number.
    We'll use a lattice arrangement and find the minimal enclosing hexagon.
    
    For N=11, a good arrangement might be:
    - A central hexagon surrounded by 6, plus 4 more arranged symmetrically.
    - Or a 3x3 grid minus 2 corners, etc.
    
    Let's try a few lattice configurations and pick the one with the smallest
    enclosing hexagon.
    """
    
    # Define several candidate lattice configurations for 11 hexagons
    # Using axial coordinates (q, r)
    
    candidates = []
    
    # Configuration 1: Linear row of 11
    for k in range(-5, 6):
        candidates.append([(k, 0)])
    # Actually, let's define full configurations
    
    # Config 1: 11 in a line
    config1 = [(k, 0) for k in range(-5, 6)]
    
    # Config 2: 3 rows: 4, 4, 3 (offset)
    config2 = []
    for r in range(-1, 2):
        if r == -1:
            for q in range(-2, 2):
                config2.append((q, r))
        elif r == 0:
            for q in range(-2, 2):
                config2.append((q, r))
        else:
            for q in range(-1, 2):
                config2.append((q, r))
    
    # Config 3: Hexagonal flower (1 center + 6 ring) + 4 extra
    # Center at (0,0), ring at distance 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # That's 7. Add 4 more at distance 2 in some directions.
    config3 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
               (2, 0), (0, 2), (-2, 2), (-2, 0)]
    
    # Config 4: 3x3 grid minus 2
    config4 = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            config4.append((q, r))
    # Remove 2 to get 11... wait 3x3=9, need 11. Let me redo.
    
    # Config 4: A 4-row arrangement: 3, 4, 3, 1
    config4 = []
    for q in range(-1, 2):
        config4.append((q, -1))
    for q in range(-2, 2):
        config4.append((q, 0))
    for q in range(-1, 2):
        config4.append((q, 1))
    config4.append((0, 2))
    
    # Config 5: Symmetric cross
    config5 = [(0, 0),
               (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
               (2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2)]
    # That's 12, remove one
    config5 = config5[:11]
    
    # Config 6: Two rows of 5 and 6
    config6 = [(q, 0) for q in range(-5, 1)] + [(q, 1) for q in range(-2, 3)]
    
    # Config 7: Compact cluster
    config7 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
               (2, -1), (-2, 1), (2, 0), (-2, 0)]
    
    all_configs = [config1, config2, config3, config4, config5, config6, config7]
    
    best_inner = None
    best_side = float('inf')
    best_angle = 0.0
    
    for config in all_configs:
        if len(config) != 11:
            continue
        
        # Convert to Cartesian
        cart_points = [hex_to_cartesian(q, r) for q, r in config]
        
        # Get all vertices of all hexagons
        all_vertices = []
        for (x, y) in cart_points:
            verts = get_hexagon_vertices(x, y, 0.0)
            all_vertices.extend(verts)
        
        all_vertices = np.array(all_vertices)
        
        # Find minimal enclosing hexagon
        side = find_min_outer_hexagon(all_vertices)
        
        if side < best_side:
            best_side = side
            best_inner = [[x, y, 0.0] for x, y in cart_points]
    
    if best_inner is None:
        # Fallback: linear arrangement
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-5, 6)]
        best_side = 40.0
    
    return best_inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
