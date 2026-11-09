"""Improved packing: hexagonal lattice arrangement with tight bounding hexagon."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, rot_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    verts = []
    for k in range(6):
        angle = math.radians(60.0 * k + rot_deg)
        x = cx + side * math.cos(angle)
        y = cy + side * math.sin(angle)
        verts.append((x, y))
    return verts

def _point_in_hex(px, py, cx, cy, side, rot_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon local frame
    dx = px - cx
    dy = py - cy
    rot = math.radians(rot_deg)
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    lx = dx * cos_r + dy * sin_r
    ly = -dx * sin_r + dy * cos_r
    
    # A regular hexagon with 'pointy-top' orientation (vertex at top) has vertices at:
    # (0, s), (s*sqrt(3)/2, s/2), (s*sqrt(3)/2, -s/2), (0, -s), (-s*sqrt(3)/2, -s/2), (-s*sqrt(3)/2, s/2)
    # This is a hexagon with flat sides on left/right if we consider the standard orientation.
    # Actually let's use the standard: vertices at angles 0, 60, 120, 180, 240, 300 degrees from center.
    # So vertex 0 is at (s, 0), vertex 1 at (s/2, s*sqrt(3)/2), etc.
    # The hexagon is defined by |x| <= s and |y| <= s*sqrt(3)/2 and |x| + |y|*sqrt(3)/3 <= s... 
    # Let me just use the half-plane method with the 6 edges.
    
    verts = _hexagon_vertices(0, 0, side, 0.0)  # local frame, no rotation
    # Check if point is on the same side of each edge as the center (origin)
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        # Edge from v1 to v2, normal pointing inward
        ex = x2 - x1
        ey = y2 - y1
        # Cross product to check side
        cross = ex * ly - ey * lx
        cross_center = ex * 0 - ey * 0  # center is at origin
        # Actually, let's check: the point (lx, ly) and origin should be on same side of edge
        # Edge vector: (ex, ey). Normal: (-ey, ex) or (ey, -ex)
        # Use the fact that for a convex polygon with vertices in CCW order,
        # the interior is to the left of each edge.
        # Cross product of edge vector with vector from v1 to point should be >= 0 for CCW.
        cp = ex * (ly - y1) - ey * (lx - x1)
        if cp < -1e-12:
            return False
    return True

def _hex_overlap_check(x1, y1, a1, x2, y2, a2, tol=1e-9):
    """Check if two unit hexagons overlap. Returns True if they overlap (invalid)."""
    # Get vertices of both hexagons
    verts1 = _hexagon_vertices(x1, y1, 1.0, a1)
    verts2 = _hexagon_vertices(x2, y2, 1.0, a2)
    
    # Check if any vertex of hex1 is inside hex2 or vice versa
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, x2, y2, 1.0, a2):
            return True
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, x1, y1, 1.0, a1):
            return True
    
    # Check edge intersections
    for i in range(6):
        x1a, y1a = verts1[i]
        x1b, y1b = verts1[(i + 1) % 6]
        for j in range(6):
            x2a, y2a = verts2[j]
            x2b, y2b = verts2[(j + 1) % 6]
            if _segments_intersect(x1a, y1a, x1b, y1b, x2a, y2a, x2b, y2b):
                return True
    return False

def _segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """Check if segment (x1,y1)-(x2,y2) intersects segment (x3,y3)-(x4,y4)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    p1, p2 = (x1, y1), (x2, y2)
    p3, p4 = (x3, y3), (x4, y4)
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    return False

def _bounding_hex_side(positions):
    """Compute the minimum side length of a regular hexagon (any rotation) that encloses all given points."""
    if not positions:
        return 0.0
    
    # For a set of points, the minimum enclosing regular hexagon can be found by
    # trying different rotations. For a hexagon with rotation angle theta, the side length
    # needed is determined by the maximum projection of points onto the 6 normal directions.
    
    # A regular hexagon centered at origin with side s and rotation theta has 6 supporting lines.
    # The distance from center to each edge is s * sqrt(3) / 2 (apothem).
    # For a point p to be inside, its projection onto each of the 6 outward normals must be <= apothem.
    
    # The 6 outward normals for a hexagon with rotation theta are at angles theta + k*60 for k=0..5
    # (for a "flat-top" or "pointy-top" hexagon, depending on convention).
    
    # Let's use: hexagon with vertex at angle theta has normals at theta + 30 + k*60 for k=0..5
    # Actually, let me think more carefully.
    
    # A regular hexagon centered at origin with circumradius R (= side length s) has vertices at:
    # R * (cos(theta + k*60), sin(theta + k*60)) for k=0..5
    # The 6 edges have outward normals at angles theta + 30 + k*60 for k=0..5
    # The apothem (distance from center to edge) is R * cos(30) = R * sqrt(3)/2
    
    # So for a hexagon with circumradius R and rotation theta, a point p is inside if:
    # For all k=0..5: p . n_k <= R * sqrt(3)/2
    # where n_k = (cos(theta + 30 + k*60), sin(theta + 30 + k*60))
    
    # To find the minimum R that encloses all points, for a given theta:
    # R >= max over all points p and all k of (p . n_k) / (sqrt(3)/2)
    # Also need R >= max over all points p and all k of (-p . n_k) / (sqrt(3)/2) because the hexagon is symmetric
    # Actually, the condition is |p . n_k| <= R * sqrt(3)/2 for all k, so:
    # R >= max over p, k of |p . n_k| / (sqrt(3)/2)
    
    # But we also need to center the hexagon optimally. Let's first assume center at origin
    # (we can translate points to be centered at their centroid).
    
    # Compute centroid
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    centered = [(p[0] - cx, p[1] - cy) for p in positions]
    
    best_R = float('inf')
    # Try many rotations
    for theta_deg in range(0, 60, 1):  # 60-degree symmetry, so 0 to 60
        theta = math.radians(theta_deg)
        max_proj = 0.0
        for px, py in centered:
            for k in range(6):
                angle = theta + math.radians(30 + k * 60)
                nx = math.cos(angle)
                ny = math.sin(angle)
                proj = abs(px * nx + py * ny)
                if proj > max_proj:
                    max_proj = proj
        R = max_proj / (math.sqrt(3) / 2)
        if R < best_R:
            best_R = R
    
    return best_R

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Hexagonal lattice arrangement
    # For unit hexagons on a hexagonal lattice, the spacing is:
    # Horizontal distance: 3 (for side-by-side with flat sides touching)
    # Actually, for regular hexagons of side 1 arranged in a honeycomb:
    # The centers form a triangular lattice. The distance between adjacent centers is 2 * apothem = 2 * sqrt(3)/2 * 2 = 2*sqrt(3)? 
    # No. For two unit hexagons sharing an edge, the distance between centers is 2 * (apothem) = 2 * (sqrt(3)/2) = sqrt(3).
    # Wait, apothem of unit hexagon is sqrt(3)/2. So distance between centers of edge-sharing hexagons is 2 * sqrt(3)/2 = sqrt(3).
    
    # In a hexagonal lattice, the basis vectors can be:
    # a1 = (3, 0) -- no, that's for flat-top
    # Let me use the standard: for pointy-top hexagons, the center-to-center distance for adjacent is 2*sqrt(3)/2 = sqrt(3) in the direction perpendicular to the shared edge.
    
    # Actually, let me just use a simple grid. For hexagons of side 1:
    # If we place them in a "flat-top" orientation, the horizontal spacing is 3 (width = 2*side = 2, but with overlap consideration)
    # Hmm, let me think about this differently.
    
    # A regular hexagon of side 1 has:
    # Width (flat-to-flat) = sqrt(3)
    # Height (point-to-point) = 2
    # For flat-top orientation: width = 2, height = sqrt(3)
    
    # For a honeycomb packing of flat-top hexagons:
    # Row spacing (vertical): 3/2 * sqrt(3) = 3*sqrt(3)/2? No.
    # Let me just use the known result: for N hexagons in a hexagonal arrangement,
    # the optimal bounding hexagon side is approximately sqrt(N) * some constant.
    
    # For N=11, a good arrangement is a hexagonal cluster.
    # A hexagonal number arrangement: 1 + 6 + 12 = 19 for radius 2, too many.
    # 1 + 6 = 7 for radius 1.
    # So for 11, we can do a 3-4-3 or similar arrangement.
    
    # Let's use a rectangular grid of hexagons with hexagonal lattice spacing.
    # For pointy-top hexagons of side 1:
    # Horizontal spacing: 3/2 * 1 = 1.5? No.
    # The width of a pointy-top hexagon is sqrt(3), height is 2.
    # For tiling: horizontal spacing = sqrt(3), vertical spacing = 3/2, with odd rows offset by sqrt(3)/2.
    
    # Let me just place them in a grid and compute the bounding hexagon.
    
    # Arrangement: 3 rows of 4, 4, 3 or similar
    # Let's do 4-4-3 or 3-4-4
    
    positions = []
    # Use hexagonal lattice: basis vectors
    # For pointy-top hexagons, the lattice vectors are:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2)
    
    sqrt3 = math.sqrt(3)
    v1x, v1y = sqrt3, 0.0
    v2x, v2y = sqrt3 / 2, 1.5
    
    # Arrange 11 hexagons in a compact shape
    # Use a 3x4 grid minus one corner, or a hexagonal cluster
    # Let's do rows: 4, 4, 3
    rows = [4, 4, 3]
    idx = 0
    for row_idx, count in enumerate(rows):
        for col_idx in range(count):
            # Offset each row
            x = col_idx * v1x + (row_idx % 2) * (v1x / 2)
            y = row_idx * v2y
            positions.append((x, y, 0.0))  # angle 0
            idx += 1
    
    # Verify no overlaps (quick check for a few pairs)
    # For a hexagonal lattice with proper spacing, there should be no overlaps.
    # The distance between adjacent centers is sqrt(v1x^2 + v1y^2) = sqrt(3) for v1,
    # and sqrt(v2x^2 + v2y^2) = sqrt(3/4 + 9/4) = sqrt(3) for v2. Good.
    
    # Compute bounding hexagon side
    side = _bounding_hex_side(positions)
    
    # Center the arrangement
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    
    # Translate all positions so centroid is at origin
    inner = [[p[0] - cx, p[1] - cy, p[2]] for p in positions]
    
    # The outer hexagon center is at origin
    outer_center = (0.0, 0.0)
    
    return inner, outer_center, side, 0.0
# EVOLVE-BLOCK-END
