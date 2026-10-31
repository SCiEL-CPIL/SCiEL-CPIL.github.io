import numpy as np
from scipy.spatial import ConvexHull
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    a = np.radians(angle_deg)
    r = side
    angles = a + np.pi/6 + np.arange(6) * np.pi/3
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def _point_in_hexagon(pt, center, side, angle_deg=0.0):
    """Check if point is inside (or on boundary of) regular hexagon."""
    a = np.radians(angle_deg)
    # Transform point to hexagon's local frame
    dx, dy = pt[0] - center[0], pt[1] - center[1]
    # Rotate by -angle
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Hexagon with flat-top: vertices at angles 0, 60, 120, 180, 240, 300 from center, radius = side
    # Actually for a regular hexagon with side length s, the distance from center to each vertex is s
    # The hexagon can be described as: |x| <= s and |y| <= s*sqrt(3)/2 and |x|/s + |y|/(s*sqrt(3)/2) <= 2/sqrt(3) ... 
    # Simpler: check against the 6 half-planes
    r = side
    # Vertices at angles 0, 60, ..., 300 (pointy-top) or 30, 90, ... (flat-top)
    # Let's use: vertices at angle k*60 degrees, radius = side
    # The hexagon is the intersection of 6 half-planes
    for k in range(6):
        theta = np.radians(60 * k)
        # The inward normal at vertex k points toward center
        # Edge between vertex k and vertex k+1
        # Midpoint of edge k-k+1:
        v1 = r * np.array([np.cos(np.radians(60*k)), np.sin(np.radians(60*k))])
        v2 = r * np.array([np.cos(np.radians(60*(k+1))), np.sin(np.radians(60*(k+1)))])
        mid = (v1 + v2) / 2
        # Normal pointing inward (toward center)
        n = -mid / np.linalg.norm(mid)
        # Point must satisfy n . (p - mid) >= 0
        if np.dot(n, [lx - mid[0], ly - mid[1]]) < -1e-9:
            return False
    return True

def _polygons_disjoint(v1, v2, tol=1e-9):
    """Check if two convex polygons (arrays of vertices) are disjoint (no interior overlap).
    They can touch at edges/vertices."""
    # Check if any vertex of v1 is strictly inside v2 or vice versa
    # Also check edge-edge intersections
    n1, n2 = len(v1), len(v2)
    
    # Point in polygon test (for convex polygon)
    def point_in_convex(pt, poly):
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i+1) % n]
            # Cross product to check if point is on the left of edge a->b
            cross = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
            if cross < -tol:
                return False
        return True
    
    # Check vertices of v1 inside v2
    for pt in v1:
        if point_in_convex(pt, v2):
            return False
    # Check vertices of v2 inside v1
    for pt in v2:
        if point_in_convex(pt, v1):
            return False
    # Check edge intersections
    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
           ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
            return True
        return False
    
    for i in range(n1):
        p1, p2 = v1[i], v1[(i+1)%n1]
        for j in range(n2):
            p3, p4 = v2[j], v2[(j+1)%n2]
            if segments_intersect(p1, p2, p3, p4):
                return False
    return True

def _min_enclosing_hexagon(points):
    """Find minimum side length regular hexagon containing all points.
    Returns (side, center, angle_deg)."""
    from scipy.optimize import minimize_scalar
    
    # The minimum enclosing regular hexagon: for a given angle, we can compute the minimum
    # side length needed to enclose all points, then optimize over angle.
    
    def side_for_angle(angle_deg):
        a = np.radians(angle_deg)
        # For a regular hexagon with center (cx,cy), side s, angle a:
        # A point (x,y) is inside iff for all 6 edges, the point is on the correct side.
        # The 6 support functions: for each of 6 directions (normals to edges),
        # the projection of the point onto that normal must be <= s * cos(30 deg) = s * sqrt(3)/2
        # Actually, for a regular hexagon with side s, the apothem (distance from center to edge) is s*sqrt(3)/2.
        # The 6 edge normals point in directions 0, 60, 120, 180, 240, 300 degrees (for pointy-top)
        # or 30, 90, ... for flat-top. Let's use directions at angles a + k*60 for k=0..5.
        
        # For each of the 6 directions, compute max projection of points
        # The hexagon centered at c with side s and orientation a contains point p iff:
        # For each k in 0..5: (p - c) . n_k <= s * sqrt(3)/2
        # where n_k = (cos(a + k*pi/3), sin(a + k*pi/3))
        # Also: (c - p) . n_k <= s * sqrt(3)/2 for the opposite... 
        # Actually the 6 half-planes are: (p-c).n_k <= s*sqrt(3)/2 for k=0..5
        # where n_k are the outward normals.
        
        # So for a given center c, the minimum s is:
        # s = max over k, over points p of (p-c).n_k / (sqrt(3)/2)
        # And we also need s >= max over k, over points p of (c-p).n_k / (sqrt(3)/2)
        # which is the same as max over k of |(p-c).n_k| / (sqrt(3)/2)
        
        # So for fixed angle and center, s = max_k max_p |(p_k - c) . n_k| / (sqrt(3)/2)
        # = (2/sqrt(3)) * max_k max_p |(p_k - c) . n_k|
        
        # To minimize s, we choose c to minimize max_k max_p |(p_k - c).n_k|
        # This is a Chebyshev-type problem.
        
        normals = np.array([[np.cos(a + k*np.pi/3), np.sin(a + k*np.pi/3)] for k in range(6)])
        
        # For each normal direction, the constraint is:
        # min_p (p.n_k) <= c.n_k <= max_p (p.n_k)
        # and s >= (max_p(p.n_k) - min_p(p.n_k)) / (2 * sqrt(3)/2) = (max-min)/sqrt(3)
        
        # Wait, let me reconsider. The hexagon with center c, side s, and edge normals n_0..n_5:
        # The region is: {p : (p-c).n_k <= s*sqrt(3)/2 for all k} AND {p : (c-p).n_k <= s*sqrt(3)/2 for all k}
        # i.e., |(p-c).n_k| <= s*sqrt(3)/2 for all k
        
        # So for all points p_i to be inside:
        # |(p_i - c).n_k| <= s*sqrt(3)/2 for all i, k
        # => s >= (2/sqrt(3)) * max_i,k |(p_i - c).n_k|
        
        # For fixed angle, minimize over c: max_i,k |(p_i - c).n_k|
        # Let q_k = c.n_k. Then (p_i - c).n_k = p_i.n_k - q_k.
        # We need: |p_i.n_k - q_k| <= M for all i,k, where M = s*sqrt(3)/2
        # For each k: q_k must be in [max_i(p_i.n_k) - M, min_i(p_i.n_k) + M]
        # This interval is non-empty iff M >= (max_i(p_i.n_k) - min_i(p_i.n_k))/2
        # So M >= max_k (max_i(p_i.n_k) - min_i(p_i.n_k))/2
        # And s = 2*M/sqrt(3) >= max_k (max_i(p_i.n_k) - min_i(p_i.n_k)) / sqrt(3)
        
        # The minimum s for a given angle is:
        # s_min = max_k (max_i(p_i.n_k) - min_i(p_i.n_k)) / sqrt(3)
        
        s_min = 0
        for k in range(6):
            proj = points.dot(normals[k])
            s_min = max(s_min, (proj.max() - proj.min()) / np.sqrt(3))
        return s_min
    
    # Optimize over angle (0 to 60 degrees by symmetry)
    result = minimize_scalar(side_for_angle, bounds=(0, 60), method='bounded', 
                             options={'xatol': 1e-6})
    best_angle = result.x
    best_side = result.fun
    
    # Find center: for the optimal angle, the center should be at the midpoint of the projections
    a = np.radians(best_angle)
    normals = np.array([[np.cos(a + k*np.pi/3), np.sin(a + k*np.pi/3)] for k in range(6)])
    
    # Center c such that c.n_k = (max_i(p_i.n_k) + min_i(p_i.n_k))/2 for each k
    # This is an overdetermined system; use least squares
    target = np.zeros(6)
    for k in range(6):
        proj = points.dot(normals[k])
        target[k] = (proj.max() + proj.min()) / 2
    
    # Solve normals.T @ c = target (6x2 system, use lstsq)
    c, _, _, _ = np.linalg.lstsq(normals.T, target, rcond=None)
    
    return best_side, (c[0], c[1]), best_angle

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Arrange 11 unit hexagons in a compact hexagonal pattern
    # Hexagonal lattice: center hex at origin, 6 around it, then 4 more
    # Actually, the densest packing of 11 hexagons is a hexagonal arrangement:
    # Row 0: 3 hexagons
    # Row 1: 5 hexagons  
    # Row 2: 3 hexagons
    # That's 11. But let's think in terms of axial coordinates.
    
    # For a hexagonal lattice with side length 1 hexagons:
    # Adjacent hexagons are 2*sqrt(3)/2 = sqrt(3) apart in the flat direction
    # Actually, for regular hexagons of side 1, the center-to-center distance
    # for edge-sharing is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)
    
    # Let's use a hexagonal grid. In axial coordinates (q, r), the position is:
    # x = sqrt(3) * (q + r/2), y = 3/2 * r  (for pointy-top hexagons)
    # or x = 3/2 * q, y = sqrt(3) * (r + q/2) for flat-top
    
    # Let me place them in a compact cluster. The minimal hexagonal number is 7 (1+6).
    # For 11, we can do a 3-4-4 arrangement or similar.
    
    # Let's try: a central hex, 6 surrounding it (total 7), then add 4 more.
    # Positions in a hex lattice (distance sqrt(3) between neighbors):
    
    # Using flat-top hexagons, neighbors are at:
    # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
    
    sqrt3 = np.sqrt(3)
    # Center at origin
    centers = [(0.0, 0.0)]
    # 6 neighbors
    for k in range(6):
        angle = np.pi/3 * k
        centers.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)))
    # That's 7. Add 4 more at distance 2*sqrt(3) in various directions
    # To keep it compact, add them adjacent to the outer ring
    # The outer ring has 6 hexagons. Adding hexagons adjacent to pairs:
    # Position at (2*sqrt(3), 0) is adjacent to (sqrt(3),0) and... let's compute
    
    # Actually, let me just use a known compact arrangement for 11 hexagons.
    # A hexagonal number H(n) = 3n(n-1)+1. H(1)=1, H(2)=7, H(3)=19.
    # 11 is between 7 and 19. A good arrangement is a "hexagon of hexagons" 
    # with radius 2 but missing some, or a different shape.
    
    # Let me try a 2-3-3-2-1 arrangement or just use a rectangular-ish grid.
    # For simplicity and compactness, let's use axial coordinates:
    
    # Axial coords (q,r) -> position: x = sqrt(3)*(q + r/2), y = 1.5*r
    # Let's pick 11 positions that form a compact shape:
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (2,0), (-2,0)
    # That's 11 positions. Let me check they're all valid hex neighbors or close.
    
    axial_positions = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (1, 1), (-1, -1),
        (2, 0), (-2, 0),
    ]
    
    inner_centers = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        inner_centers.append((x, y))
    
    # All hexagons have the same orientation (0 degrees) for simplicity
    # Each unit hexagon has side 1, angle 0
    inner = [[cx, cy, 0.0] for cx, cy in inner_centers]
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for cx, cy, ang in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_vertices.extend(verts.tolist())
    all_vertices = np.array(all_vertices)
    
    # Find minimum enclosing regular hexagon
    side, center, angle = _min_enclosing_hexagon(all_vertices)
    
    # Verify the solution is valid (no overlaps, all inside)
    # Check no overlaps between inner hexagons
    valid = True
    hex_verts = [_hex_vertices(cx, cy, 1.0, ang) for cx, cy, ang in inner]
    for i in range(len(hex_verts)):
        for j in range(i+1, len(hex_verts)):
            if not _polygons_disjoint(hex_verts[i], hex_verts[j]):
                valid = False
                break
        if not valid:
            break
    
    # Check all inner hexagons are inside the outer hexagon
    if valid:
        for i, (cx, cy, ang) in enumerate(inner):
            verts = hex_verts[i]
            for v in verts:
                if not _point_in_hexagon(v, center, side, angle):
                    valid = False
                    break
            if not valid:
                break
    
    if valid:
        return inner, center, side, angle
    else:
        # Fallback: use a larger hexagon
        # Compute bounding box and use a generous outer hexagon
        min_x, min_y = all_vertices.min(axis=0)
        max_x, max_y = all_vertices.max(axis=0)
        center = ((min_x + max_x)/2, (min_y + max_y)/2)
        # Use a hexagon that definitely contains everything
        max_dist = np.max(np.sqrt((all_vertices[:,0]-center[0])**2 + (all_vertices[:,1]-center[1])**2))
        side = max_dist * 2  # Definitely large enough
        return inner, center, side, 0.0
# EVOLVE-BLOCK-END
