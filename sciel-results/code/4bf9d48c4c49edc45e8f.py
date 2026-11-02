"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    # Ensure side is positive
    side = abs(side)
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees.
    Vertices are at angles 0, 60, 120, 180, 240, 300 degrees (flat-top when angle=0)."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    # For a regular hexagon with side s, the vertices are at distance s from center
    # The hexagon in local coords (flat-top, angle=0) has vertices at:
    # (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # It can be described as the intersection of 6 half-planes.
    # For a hexagon with "radius" (center to vertex) = s:
    # The 6 supporting lines are at distance s/2 from center along directions 30, 90, 150, 210, 270, 330
    # Actually for a regular hexagon with side length s, the apothem (center to side midpoint) is s*sqrt(3)/2
    # and the circumradius (center to vertex) is s.
    # The 6 sides are perpendicular to directions 0, 60, 120, 180, 240, 300 at distance apothem.
    # Wait, let me think more carefully.
    # Vertices at angles 0, 60, 120, 180, 240, 300 at distance s.
    # Side midpoints at angles 30, 90, 150, 210, 270, 330 at distance s*sqrt(3)/2.
    # The half-plane for each side: dot(point, normal) <= apothem
    # where normal is the unit vector from center to side midpoint.
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons are disjoint (non-overlapping interiors).
    hex1, hex2: (cx, cy, side, angle_deg)
    We check if any vertex of one is inside the other, and vice versa.
    Also check edge intersections for robustness, but vertex-in-polygon is usually sufficient
    for convex polygons if no vertex is inside either."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    
    # Quick bounding box check
    r1 = s1  # circumradius
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx*dx + dy*dy > (r1 + r2)**2 + 1e-12:
        return True  # definitely disjoint
    
    # Check if any vertex of hex1 is inside hex2
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    
    # Check if any vertex of hex2 is inside hex1
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    
    # For convex polygons, if no vertex of either is inside the other, they may still overlap
    # if edges cross. Let's do a simple edge intersection check.
    # Actually, for two convex polygons, they are disjoint iff no vertex of one is in the other
    # AND no edges cross. But if no vertex is in the other, edges can still cross.
    # Let's add edge crossing check.
    
    def _segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4 (proper intersection or touching)."""
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
    
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    
    return True

def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _point_in_outer_hex(px, py, ox, oy, oside, oangle):
    """Check if point is inside the outer hexagon."""
    return _point_in_hex(px, py, ox, oy, oside, oangle)

def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Hexagonal lattice packing
    # For unit hexagons in a hexagonal lattice:
    # The centers form a triangular lattice.
    # Distance between adjacent centers = 2 * apothem = sqrt(3) for side-1 hexagons (touching)
    # Actually, for two unit hexagons to be disjoint but touching, center distance = 2 * apothem = sqrt(3)
    # Wait, apothem of unit hexagon = sqrt(3)/2, so 2*apothem = sqrt(3)
    
    # Let's use a hexagonal arrangement of N hexagons.
    # For N hexagons arranged in a hexagonal pattern (like a hex number),
    # we need to figure out the arrangement.
    
    # Simple approach: place them in a hexagonal grid pattern.
    # For a hexagonal grid, use axial coordinates (q, r).
    # Position: x = (q + r/2) * sqrt(3), y = r * 1.5
    # (for flat-top hexagons with side 1, the distance between centers in the grid)
    
    # Actually, let me think about this more carefully.
    # For regular hexagons of side 1 in a hexagonal tiling:
    # The centers are at distance sqrt(3) apart for adjacent hexagons.
    # In a hex grid with axial coords (q, r):
    #   x = sqrt(3) * (q + r/2)
    #   y = 1.5 * r
    
    # We want to place N hexagons in a compact hexagonal shape.
    # Hex numbers: 1, 7, 19, 37, ... = 3k(k+1)+1 for k=0,1,2,3...
    # For N=11, it's not a perfect hex number.
    
    # Let's just place them in a hexagonal grid pattern centered at origin.
    # Use axial coordinates, starting from center.
    
    def hex_to_pixel(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    # Generate hex grid positions in order of distance from center
    def generate_hex_positions(n):
        """Generate n positions in a hex grid, starting from center, expanding outward."""
        if n <= 1:
            return [(0, 0)] if n == 1 else []
        
        positions = [(0, 0)]
        seen = {(0, 0)}
        # Ring 1: 6 positions
        ring = 1
        while len(positions) < n:
            # Generate ring `ring`
            # In axial coordinates, ring k consists of positions at hex distance k from origin
            # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
            dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
            # Start position
            sq, sr = 0, 0
            for di, (dq, dr) in enumerate(dirs):
                # Move in this direction for `ring` steps
                for step in range(ring):
                    sq += dq
                    sr += dr
                    if (sq, sr) not in seen:
                        seen.add((sq, sr))
                        positions.append((sq, sr))
                        if len(positions) >= n:
                            break
                if len(positions) >= n:
                    break
            ring += 1
        return positions[:n]
    
    # Try hexagonal lattice packing
    positions = generate_hex_positions(N)
    inner_hexes = []
    # Increase spacing to ensure disjointness (gap factor 1.05)
    scale = math.sqrt(3) * 1.05
    for q, r in positions:
        x, y = hex_to_pixel(q, r)
        x *= scale
        y *= scale
        inner_hexes.append((x, y, 1.0, 0.0))
    
    # Check if they're all disjoint
    if not _all_disjoint(inner_hexes):
        # Fallback: use a simpler arrangement with more spacing
        # Place in a line with sufficient spacing
        spacing = math.sqrt(3) + 0.2  # significantly more than touching
        inner_hexes = []
        for k in range(-(N // 2), N - N // 2):
            inner_hexes.append((k * spacing, 0.0, 1.0, 0.0))
    
    # Compute minimum outer hexagon side length needed
    # The outer hexagon must contain all inner hexagons.
    # For a hexagon containing other hexagons, we need to find the minimum side length.
    
    # Simple approach: find the bounding extent of all inner hexagon vertices
    # and compute the minimum enclosing hexagon side length.
    
    # For a regular hexagon centered at origin with side S, the constraint for a point (px, py)
    # to be inside is: for each of 6 directions, the projection must be <= S*sqrt(3)/2
    # (the apothem of the outer hexagon)
    
    # Actually, let me compute it differently.
    # For a point (px, py) to be inside a hexagon centered at (ox, oy) with side S and angle 0:
    # The hexagon has vertices at distance S from center at angles 0, 60, 120, 180, 240, 300.
    # The 6 sides are at distance S*sqrt(3)/2 from center in directions 30, 90, 150, 210, 270, 330.
    # So the constraint is: for each direction theta_k = 30 + 60*k degrees,
    #   (px - ox) * cos(theta_k) + (py - oy) * sin(theta_k) <= S * sqrt(3) / 2
    
    # To find minimum S for center (ox, oy) and angle 0:
    # S >= max over all points and all k of [ (px - ox) * cos(theta_k) + (py - oy) * sin(theta_k) ] * 2 / sqrt(3)
    
    def min_outer_side(hexes, ox, oy, oangle=0.0):
        """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
        that contains all hexagons in `hexes`."""
        max_val = 0.0
        a = -math.radians(oangle)
        for cx, cy, s, ang in hexes:
            verts = _hex_vertices(cx, cy, s, ang)
            for vx, vy in verts:
                # Transform to outer hex frame
                dx = vx - ox
                dy = vy - oy
                lx = dx * math.cos(a) - dy * math.sin(a)
                ly = dx * math.sin(a) + dy * math.cos(a)
                for k in range(6):
                    theta = math.radians(30 + 60 * k)
                    proj = lx * math.cos(theta) + ly * math.sin(theta)
                    if proj > max_val:
                        max_val = proj
        return max_val * 2.0 / math.sqrt(3)
    
    # Try different outer hexagon centers and angles to minimize the side length
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # The optimal outer hexagon center is likely near the centroid of the inner hexagons
    # Let's try the centroid and a few nearby points
    cx_mean = sum(h[0] for h in inner_hexes) / len(inner_hexes)
    cy_mean = sum(h[1] for h in inner_hexes) / len(inner_hexes)
    
    # Try several candidate centers
    candidates = [(cx_mean, cy_mean), (0.0, 0.0)]
    # Add some offset candidates
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            candidates.append((cx_mean + dx * 0.5, cy_mean + dy * 0.5))
    
    # Try several angles
    angles = [0.0, 30.0, 60.0, 15.0, 45.0, 75.0]
    
    for ocx, ocy in candidates:
        for oang in angles:
            if time.time() > deadline:
                break
            side = min_outer_side(inner_hexes, ocx, ocy, oang)
            if side < best_side:
                best_side = side
                best_center = (ocx, ocy)
                best_angle = oang
    
    # Verify the solution
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break
    
    if not valid:
        # Increase side length slightly
        best_side *= 1.01
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    
    # Format output
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]  # (x, y, angle_degrees)
    return inner, (best_center[0], best_center[1]), best_side, best_angle

# EVOLVE-BLOCK-END
