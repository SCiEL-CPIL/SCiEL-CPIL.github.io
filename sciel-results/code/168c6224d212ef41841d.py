"""Packing 12 unit hexagons inside a minimal regular hexagon."""
import math
import numpy as np

N = 12

def hex_vertices(cx, cy, s, rot_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation rot_deg."""
    rot = math.radians(rot_deg)
    pts = []
    for i in range(6):
        a = rot + i * math.pi / 3.0
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return np.array(pts)

def hex_corner_dist(cx, cy, s, rot_deg):
    """Max distance from center to any vertex of the hexagon (circumradius = s for regular hex)."""
    return s  # regular hexagon circumradius equals side length

def polygon_collision(p1, p2):
    """Check if two convex polygons (arrays of (x,y)) intersect (including boundary touch)."""
    # Separating axis theorem for convex polygons
    def axes(poly):
        ax = []
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            # normal
            nx, ny = -dy, dx
            L = math.hypot(nx, ny)
            if L > 1e-12:
                ax.append((nx / L, ny / L))
        return ax

    def project(poly, axis):
        vals = [x * axis[0] + y * axis[1] for x, y in poly]
        return min(vals), max(vals)

    for axis in axes(p1) + axes(p2):
        min1, max1 = project(p1, axis)
        min2, max2 = project(p2, axis)
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
    return True

def point_in_hexagon(px, py, cx, cy, s, rot_deg):
    """Check if point (px,py) is inside the hexagon."""
    rot = math.radians(rot_deg)
    # Transform point to hexagon's local frame
    dx, dy = px - cx, py - cy
    lx = dx * math.cos(-rot) - dy * math.sin(-rot)
    ly = dx * math.sin(-rot) + dy * math.cos(-rot)
    # Regular hexagon with flat-top or pointy-top? Using standard: vertices at angles 0,60,120...
    # For a regular hexagon with circumradius s, the inscribed radius (apothem) is s*sqrt(3)/2
    # In local frame, hexagon is defined by |lx| <= s and |ly| <= s*sqrt(3)/2 and combinations
    # Actually let's just check against the 6 half-planes
    # Vertices in local frame (no rotation): (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # Edges connect consecutive vertices; check point is on the correct side of each edge
    verts = []
    for i in range(6):
        a = i * math.pi / 3.0
        verts.append((s * math.cos(a), s * math.sin(a)))
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        # Cross product to check which side
        ex, ey = x2 - x1, y2 - y1
        px_local, py_local = lx - x1, ly - y1
        cross = ex * py_local - ey * px_local
        # The center (0,0) should be on the same side as interior
        cx_local, cy_local = -x1, -y1
        cross_c = ex * cy_local - ey * cx_local
        if cross * cross_c < -1e-9:
            return False
    return True

def hex_in_outer(cx, cy, s_inner, rot_inner, outer_cx, outer_cy, s_outer, rot_outer):
    """Check if inner hexagon is fully inside outer hexagon."""
    inner_verts = hex_vertices(cx, cy, s_inner, rot_inner)
    for vx, vy in inner_verts:
        if not point_in_hexagon(vx, vy, outer_cx, outer_cy, s_outer, rot_outer):
            return False
    return True

def construct():
    # Strategy: place 12 unit hexagons in a compact arrangement, then fit inside outer hexagon.
    # Use a 2-row or 3-row honeycomb-like layout, then compute minimal enclosing hexagon.
    # 
    # A regular hexagon of side 1 has width (flat-to-flat) = sqrt(3), point-to-point = 2.
    # Let's try a layout: 4 hexagons in a row, 4 in next row offset, 4 in next row.
    # Or better: use a hexagonal packing pattern.
    
    # Let's try: arrange in a roughly circular/hexagonal cluster.
    # For 12 hexagons, a natural arrangement is a hexagon of side 2 (which has 1+6+12=19 hexagons)
    # but we only need 12. Let's use a 4-4-4 rectangular-ish grid with offsets.
    
    # Flat-top hexagons: width = 2*s, height = sqrt(3)*s
    # For side=1: width=2, height=sqrt(3)≈1.732
    # Horizontal spacing: 1.5*s = 1.5
    # Vertical spacing: sqrt(3)*s = sqrt(3)
    # Offset rows by 0.5*s = 0.5 horizontally
    
    s = 1.0
    positions = []
    # Row 0: 4 hexagons
    for i in range(4):
        x = i * 1.5
        y = 0.0
        positions.append((x, y, 0.0))
    # Row 1: 4 hexagons, offset
    for i in range(4):
        x = i * 1.5 + 0.75
        y = math.sqrt(3)
        positions.append((x, y, 0.0))
    # Row 2: 4 hexagons
    for i in range(4):
        x = i * 1.5
        y = 2 * math.sqrt(3)
        positions.append((x, y, 0.0))
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (min(xs) + max(xs)) / 2.0
    cy_mean = (min(ys) + max(ys)) / 2.0
    
    inner = []
    for x, y, ang in positions:
        inner.append([x - cx_mean, y - cy_mean, ang])
    
    # Now compute the minimal outer hexagon that contains all inner hexagons.
    # The outer hexagon has center (0,0) and we need to find its side length.
    # For a regular hexagon with center at origin and rotation 0, the boundary is:
    # |x| <= s, |y| <= s*sqrt(3)/2, and |x|/2 + |y|/sqrt(3) <= s (approximately)
    # Actually, the hexagon with vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
    # The constraint for a point (x,y) to be inside is:
    # max(|x|, |x|/2 + |y|*sqrt(3)/2, |-x|/2 + |y|*sqrt(3)/2) <= s ... let me think more carefully.
    
    # For a regular hexagon centered at origin with "pointy-top" orientation (vertex at top),
    # vertices at angles 90, 150, 210, 270, 330, 30 degrees.
    # Let's use flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees.
    # Vertex i: (s*cos(i*pi/3), s*sin(i*pi/3))
    # The hexagon is the intersection of 6 half-planes.
    # For edge between vertex i and vertex i+1, the inward normal points toward center.
    
    # For flat-top hexagon with circumradius s (side length = s for regular hex):
    # The 6 edges have normals at angles 30, 90, 150, 210, 270, 330 degrees (perpendicular to edges).
    # Edge between vertex at 0° and 60°: midpoint direction is 30°, normal is 30° (inward).
    # Distance from center to each edge (apothem) = s * cos(30°) = s * sqrt(3)/2.
    
    # So a point (x,y) is inside iff for all 6 normals n_i: dot((x,y), n_i) <= apothem
    # where n_i are unit vectors at 30, 90, 150, 210, 270, 330 degrees.
    
    normals = []
    for i in range(6):
        a = math.radians(30 + i * 60)
        normals.append((math.cos(a), math.sin(a)))
    
    # For each inner hexagon, find the max dot product with each normal over its vertices.
    # The outer hexagon side length s_outer must satisfy: apothem_outer >= max_dot
    # where apothem_outer = s_outer * sqrt(3)/2
    
    max_dot = 0.0
    for ix, iy, iang in inner:
        verts = hex_vertices(ix, iy, 1.0, iang)
        for vx, vy in verts:
            for nx, ny in normals:
                d = vx * nx + vy * ny
                if d > max_dot:
                    max_dot = d
    
    # s_outer * sqrt(3)/2 >= max_dot
    # s_outer >= max_dot * 2 / sqrt(3)
    s_outer = max_dot * 2.0 / math.sqrt(3)
    
    # Add small margin for safety
    s_outer *= 1.001
    
    return inner, (0.0, 0.0), s_outer, 0.0
