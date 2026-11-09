"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START

def _hexagon_vertices(cx, cy, side, rot_deg):
    """
    Return 6 vertices of a regular hexagon centered at (cx,cy) with given side
    and rotation in degrees. The hexagon is 'pointy-top' at rot=0:
    vertices at angles 90, 150, 210, 270, 330, 30 degrees (i.e., starting from top).
    Actually, let's define it so that at rot=0, the vertices are at angles 0,60,120,180,240,300
    (flat-top? or pointy-top?).
    Standard regular hexagon with side length s has circumradius s.
    Let's define vertices at angles 0, 60, 120, 180, 240, 300 degrees relative to center.
    This gives a 'pointy-right' hexagon? No, at 0 degrees it's (s, 0).
    Let's use the standard orientation where the 'width' (vertex to vertex) is 2*s.
    For a unit hexagon, vertices at angles 0, 60, ... are at distance 1 from center.
    """
    angles = np.deg2rad(rot_deg) + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def _outer_hexagon_vertices(cx, cy, side, rot_deg):
    """
    Return 6 vertices of the outer regular hexagon.
    Same orientation as inner: vertices at 0, 60, ... + rot_deg.
    """
    return _hexagon_vertices(cx, cy, side, rot_deg)

def _point_in_convex_polygon(p, poly):
    """
    Check if point p is inside convex polygon poly (CCW or CW).
    We assume poly is given in CCW order.
    For a regular hexagon defined as above, is it CCW?
    Angles 0, 60, 120... increases, so CCW.
    """
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        edge = p2 - p1
        vec = p - p1
        # Cross product in 2D
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        # For CCW polygon, all points inside should have cross >= 0
        # (assuming origin is inside, or generally consistent sign)
        if cross < -1e-9:
            return False
    return True

def _hexagons_intersect(h1_center, h1_side, h1_rot, h2_center, h2_side, h2_rot, tol=1e-6):
    """
    Check if two regular hexagons intersect (overlap area > tol).
    Simple check: if distance between centers > sum of circumradii, they are disjoint.
    Circumradius of regular hexagon is its side length.
    So if dist > h1_side + h2_side + tol, they are disjoint.
    But this is a sufficient condition for disjointness, not necessary.
    They can be closer and still disjoint if rotated.
    For unit hexagons, max distance between centers for disjointness depends on orientation.
    The 'width' (vertex to vertex) is 2*s.
    The 'height' (side to side) is s*sqrt(3).
    
    A more robust check: check if any vertex of h1 is inside h2 or vice versa,
    or if any edge of h1 intersects any edge of h2.
    Since we are packing tightly, let's use a slightly conservative distance check first,
    then do a precise check if needed.
    
    Actually, for the purpose of this task, we can just place them with some spacing
    to ensure they are disjoint, and then shrink the outer hexagon.
    
    Let's implement a precise intersection check.
    """
    d = np.linalg.norm(np.array(h1_center) - np.array(h2_center))
    # Quick reject
    if d > h1_side + h2_side + tol:
        return False
    
    v1 = _hexagon_vertices(h1_center[0], h1_center[1], h1_side, h1_rot)
    v2 = _hexagon_vertices(h2_center[0], h2_center[1], h2_side, h2_rot)
    
    # Check if any vertex of v1 is inside v2
    for p in v1:
        if _point_in_convex_polygon(p, v2):
            return True
    # Check if any vertex of v2 is inside v1
    for p in v2:
        if _point_in_convex_polygon(p, v1):
            return True
            
    # Check edge intersections
    for i in range(6):
        e1 = [v1[i], v1[(i+1)%6]]
        for j in range(6):
            e2 = [v2[j], v2[(j+1)%6]]
            if _segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True
    return False

def _segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects p3-p4."""
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

def construct():
    # We want to pack N=11 unit hexagons.
    # Let's try a hexagonal grid arrangement.
    # Positions for a hex grid (pointy-top or flat-top?).
    # Let's use flat-top hexagons for the grid? Or just place them on a Cartesian grid with appropriate spacing.
    
    # Spacing for flat-top hexagons:
    # Horizontal distance between centers: 1.5 * side
    # Vertical distance: sqrt(3) * side
    # Alternating columns are offset by sqrt(3)/2 * side.
    
    # Let's use flat-top hexagons (rot = 0 in our definition? No, our definition at rot=0 has vertex at 0 deg, so it's pointy-right/flat-top?
    # Vertex at (1,0). Next at (0.5, 0.866).
    # This is a flat-top hexagon? No, a flat-top hexagon has a horizontal top edge.
    # Vertices at 30, 90, 150, 210, 270, 330 would be pointy-top?
    # Let's just use the standard grid coordinates for hexagons.
    
    # Let's try to place them in a 'flower' pattern: 1 center, 6 around, 4 more.
    # Center at (0,0).
    # Ring 1: 6 hexagons at distance 2*side * cos(30) = sqrt(3) * side?
    # No, distance between centers of touching hexagons is 2*side (vertex to vertex) or sqrt(3)*side (edge to edge)?
    # If they touch edge-to-edge, distance is sqrt(3)*side.
    # If they touch vertex-to-vertex, distance is 2*side.
    # For minimal area, we want them as close as possible without overlapping.
    # Closest distance between centers of two unit hexagons without overlap is 2 * inradius = sqrt(3) * side?
    # Inradius (apothem) of unit hexagon is sqrt(3)/2.
    # So if two hexagons touch edge-to-edge, distance between centers is 2 * sqrt(3)/2 = sqrt(3).
    # So let's place them on a triangular lattice with spacing sqrt(3).
    
    s = 1.0
    dist = s * math.sqrt(3)
    
    # Generate positions
    # Center
    positions = [(0.0, 0.0)]
    
    # Ring 1: 6 positions
    for i in range(6):
        angle = math.pi / 3 * i
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        positions.append((x, y))
        
    # Ring 2: We need 4 more.
    # Ring 2 positions are at distance 2*dist from center, or at distance dist from ring 1 positions.
    # Let's pick 4 positions from the second ring of the triangular lattice.
    # The second ring has 12 positions.
    # We can pick 4 that are 'farthest' from each other to balance the shape, or just pick any 4.
    # To minimize the bounding hexagon, we should pick them symmetrically.
    # Let's pick 4 at angles 0, 90, 180, 270? No, the lattice is 6-fold symmetric.
    # Let's pick 4 consecutive ones? Or 2 opposite pairs?
    # Let's pick positions at angles 0, 60, 180, 240?
    # Actually, just picking any 4 will work for validity. For optimal score, we need to find the minimal enclosing hexagon.
    # Let's pick 4 positions that are far from the center.
    # The 12 positions in ring 2 are at distance 2*dist from center.
    # Let's pick 4 of them.
    
    # List of ring 2 positions
    ring2 = []
    for i in range(12):
        angle = math.pi / 6 * i
        x = 2 * dist * math.cos(angle)
        y = 2 * dist * math.sin(angle)
        ring2.append((x, y))
        
    # Pick 4 positions. Let's pick every 3rd one to spread them out.
    # Indices 0, 3, 6, 9
    indices = [0, 3, 6, 9]
    for idx in indices:
        positions.append(ring2[idx])
        
    # Total 1 + 6 + 4 = 11 positions.
    
    # Rotate all inner hexagons to 0 degrees (flat-top? or pointy-top?)
    # Our definition at rot=0 has vertex at (1,0).
    # Let's keep them all at 0 degrees.
    
    inner = []
    for p in positions:
        inner.append([p[0], p[1], 0.0])
        
    # Now find the minimal enclosing regular hexagon.
    # The outer hexagon can be rotated.
    # We need to find the smallest side length S such that all inner hexagons are inside the outer hexagon.
    # Outer hexagon center can also move? The task says "outer hexagon center (x, y), side length and rotation".
    # So we can optimize center, side, and rotation.
    
    # Let's fix the outer hexagon center at (0,0) for now, and vary rotation.
    # Then for each rotation, find the minimum S such that all inner hexagons are inside.
    # To check if an inner hexagon is inside the outer hexagon, we can check if all vertices of the inner hexagon
    # are inside the outer hexagon. Since both are convex, this is sufficient.
    
    def min_outer_side(rotation_deg):
        # Compute vertices of outer hexagon with side S, center (0,0), rotation rotation_deg.
        # We need to find minimal S.
        # For a given rotation, the constraint for each inner hexagon vertex (vx, vy) is that it must be inside the outer hexagon.
        # The outer hexagon is defined by 6 inequalities.
        # For a regular hexagon with center (0,0), side S, rotation r:
        # The distance from origin to each side is S * sqrt(3) / 2.
        # The normals to the sides are at angles r, r+60, ..., r+300.
        # A point (x,y) is inside if x*cos(n) + y*sin(n) <= S*sqrt(3)/2 for all 6 normals n.
        # So for each inner hexagon vertex, we compute the max of (x*cos(n_i) + y*sin(n_i)) over i.
        # This max must be <= S*sqrt(3)/2.
        # So S >= max_val / (sqrt(3)/2) for all vertices.
        # Thus minimal S is max over all vertices of (max_i (x*cos(n_i) + y*sin(n_i))) / (sqrt(3)/2).
        
        normals = np.deg2rad(rotation_deg) + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
        max_projection = 0.0
        
        for ix, iy, irot in inner:
            # Compute vertices of this inner hexagon
            verts = _hexagon_vertices(ix, iy, 1.0, irot)
            for v in verts:
                proj = v[0] * np.cos(normals) + v[1] * np.sin(normals)
                m = np.max(proj)
                if m > max_projection:
                    max_projection = m
                    
        S = max_projection / (math.sqrt(3) / 2.0)
        return S
        
    # Optimize rotation
    best_S = float('inf')
    best_rot = 0.0
    # Try rotations in small steps
    for r in np.linspace(0, 60, 61):
        S = min_outer_side(r)
        if S < best_S:
            best_S = S
            best_rot = r
            
    # We can also optimize the center?
    # If we shift the outer hexagon center, the projections change.
    # Let's assume the optimal center is the centroid of the inner hexagons, or (0,0).
    # Given the symmetry of our packing, (0,0) is likely optimal.
    
    return inner, (0.0, 0.0), best_S, best_rot

# EVOLVE-BLOCK-END
