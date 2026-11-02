"""Improved program: lattice-based construction with optimization."""
import math
import time
import itertools

N = 12
DEADLINE = 130.0  # seconds, safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _hexagon_support(cx, cy, side, angle_deg):
    """For a hexagon, return the 6 outward normal directions (unit vectors) and the support function values."""
    # A regular hexagon with 'flat' orientation (angle 0) has normals at 0, 60, 120, 180, 240, 300 degrees
    # Actually, let's define: a regular hexagon with vertices at angles 0, 60, ... has faces with normals at 30, 90, 150, ...
    # Let me think more carefully.
    # A regular hexagon centered at origin with side s, rotated by angle a.
    # The support function h(n) = max over vertices of v·n.
    # For a regular hexagon, the support in direction n equals s * max over the 6 edge normals of (n · e_i) where e_i are unit edge normals, but actually it's simpler:
    # The hexagon is the intersection of 6 half-planes. The support in direction n is the max of the dot products of the 6 vertices with n.
    # For optimization, we need: for the outer hexagon to contain an inner hexagon, for every direction n (unit), the support of outer in direction n >= support of inner in direction n.
    # For a regular hexagon, the support function is piecewise linear in n, with 6 linear pieces.
    # Actually, for two convex polygons, one contains the other iff for every edge normal of the inner, the support of outer >= support of inner, AND for every edge normal of outer... no.
    # Simpler: polygon A contains polygon B iff for every edge of B, the half-plane defined by that edge (containing B) also contains all of A... no, that's the opposite.
    # A contains B iff for every vertex of B, it's inside A. For convex polygons, it suffices to check that all vertices of B are inside A.
    # So: inner hexagon is inside outer hexagon iff all 6 vertices of inner are inside outer.
    # Two inner hexagons are disjoint iff no edge of one intersects any edge of the other, and neither contains the other.
    # For regular hexagons of the same size, disjointness is equivalent to: the minimum distance between any two vertices/edges >= 0, which is equivalent to: for all pairs of vertices (one from each), they don't cross edges.
    # Actually, for two congruent regular hexagons, they are disjoint iff the distance between their centers is >= some threshold depending on relative rotation, OR they overlap.
    # The cleanest check: for two convex polygons, they are disjoint iff there exists a separating line. For regular hexagons, we can check all pairs of edges.

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    # Transform to local coordinates
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    # Rotate point by -a
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # In local coords, hexagon with flat top (vertices at 0, 60, ...) has:
    # |lx| <= side and |ly| <= side * sqrt(3)/2 and |ly| <= sqrt(3) * (side - |lx|/2) ... 
    # Actually, for a regular hexagon with vertices at (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2):
    # The boundary is: |ly| <= sqrt(3)/2 * s for |lx| <= s/2, and |ly| <= sqrt(3) * (s - |lx|) for s/2 <= |lx| <= s
    # Let me verify: at lx = s/2, ly = sqrt(3)/2 * s. At lx = s, ly = 0. The line from (s/2, sqrt(3)/2*s) to (s, 0) has slope (0 - sqrt(3)/2*s)/(s - s/2) = -sqrt(3).
    # So ly = -sqrt(3)*(lx - s) = sqrt(3)*(s - lx) for lx in [s/2, s]. Yes.
    sqrt3 = math.sqrt(3)
    abs_lx = abs(lx)
    abs_ly = abs(ly)
    if abs_lx <= side / 2:
        return abs_ly <= sqrt3 / 2 * side
    elif abs_lx <= side:
        return abs_ly <= sqrt3 * (side - abs_lx)
    else:
        return False

def _hexagons_disjoint(cx1, cy1, a1, cx2, cy2, a2, s=1.0):
    """Check if two unit hexagons are disjoint (no interior overlap)."""
    # Get vertices of both
    v1 = _hex_vertices(cx1, cy1, s, a1)
    v2 = _hex_vertices(cx2, cy2, s, a2)
    # Check if any vertex of one is inside the other
    for vx, vy in v1:
        if _point_in_hexagon(vx, vy, cx2, cy2, s, a2):
            return False
    for vx, vy in v2:
        if _point_in_hexagon(vx, vy, cx1, cy1, s, a1):
            return False
    # Check edge intersections
    def _segments_intersect(p1, p2, p3, p4):
        def _cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = _cross(p3, p4, p1)
        d2 = _cross(p3, p4, p2)
        d3 = _cross(p1, p2, p3)
        d4 = _cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    return True

def _outer_contains_inner(outer_cx, outer_cy, outer_side, outer_angle, inner_cx, inner_cy, inner_angle, inner_side=1.0):
    """Check if inner hexagon is inside outer hexagon."""
    inner_verts = _hex_vertices(inner_cx, inner_cy, inner_side, inner_angle)
    for vx, vy in inner_verts:
        if not _point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def _min_outer_side(inner_list, outer_cx, outer_cy, outer_angle):
    """Given inner hexagons and outer center/angle, find minimum outer side that contains all."""
    # For a regular hexagon with given center and angle, the minimum side to contain a set of points
    # is determined by the maximum over all 6 face directions of the support.
    # For a hexagon with center (cx,cy), side s, angle a, the 6 face normals are at angles a+30, a+90, a+150, a+210, a+270, a+330 (if vertices at a, a+60, ...)
    # Actually, let me reconsider. If vertices are at angles a, a+60, ..., then the edges are between consecutive vertices.
    # The outward normal of the edge between vertex at angle a and vertex at angle a+60 points in direction a+30.
    # The distance from center to that edge is s * cos(30) = s * sqrt(3)/2.
    # So the support in direction n (unit) is: for the hexagon, h(n) = s * sqrt(3)/2 * max over 6 normals of (n · e_i) where e_i are the 6 unit edge normals.
    # Wait, no. The support function of a regular hexagon: h(n) = s * cos(30°) / cos(angle between n and nearest edge normal) ... this is getting complicated.
    # Simpler: for a given direction n, the support of the hexagon is the max of v·n over its 6 vertices.
    # The minimum s such that all points p are inside is: s = max over all points p and all 6 face directions of (p · n_i - c · n_i) / (sqrt(3)/2)
    # where n_i are the 6 unit edge normals.
    # Let me just compute: for each of the 6 edge normals, compute the max projection of all inner vertices onto that normal, minus the center's projection, and divide by (sqrt(3)/2).
    a = math.radians(outer_angle)
    sqrt3 = math.sqrt(3)
    max_ratio = 0.0
    for i in range(6):
        # Edge normal direction: the edge between vertex i and vertex i+1 has normal at angle a + 30 + 60*i
        n_angle = a + math.radians(30 + 60 * i)
        nx = math.cos(n_angle)
        ny = math.sin(n_angle)
        # Support of inner hexagons in this direction
        max_proj = float('-inf')
        for (ix, iy, ia) in inner_list:
            iverts = _hex_vertices(ix, iy, 1.0, ia)
            for (vx, vy) in iverts:
                proj = (vx - outer_cx) * nx + (vy - outer_cy) * ny
                if proj > max_proj:
                    max_proj = proj
        # The hexagon with side s has support s*sqrt(3)/2 in its edge normal direction
        # So we need s*sqrt(3)/2 >= max_proj, i.e., s >= max_proj / (sqrt(3)/2)
        ratio = max_proj / (sqrt3 / 2)
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio

def construct():
    start_time = time.time()
    
    # Strategy: Place 12 unit hexagons in a compact cluster, then optimize outer hexagon.
    # Use a known good configuration: a "flower" or hexagonal packing.
    
    # Let's try a configuration where hexagons are arranged in a hexagonal lattice pattern.
    # For 12 hexagons, a good arrangement might be:
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons  
    # Row 3: 4 hexagons
    # But with hexagonal geometry, rows are offset.
    
    # Let me try: a central hexagon surrounded by others.
    # Actually, for regular hexagons, the densest packing is hexagonal.
    
    # Let's try a 3-4-3-2 arrangement or similar.
    
    # Simple approach: place them in a grid-like pattern with hexagonal offsets.
    # Hexagonal lattice: spacing between centers in adjacent rows is sqrt(3) for same orientation.
    
    # Let's try: 4 rows of 3, with hexagonal offset.
    # Row 0: 3 hexagons at x = -1, 0, 1 (with spacing 2*sqrt(3)/2 = sqrt(3) horizontally? No)
    
    # For unit hexagons with flat top, the horizontal distance between adjacent centers in the same row is 2 * side * cos(30) = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732.
    # The vertical distance between rows is 2 * side * sin(30) = 1.
    # Wait, let me reconsider. For a regular hexagon with side 1, flat-top orientation:
    # Width (horizontal) = 2 * side = 2
    # Height (vertical) = sqrt(3) * side = sqrt(3)
    # No wait. If vertices are at 0, 60, 120, ... then:
    # Vertex at 0: (1, 0), vertex at 60: (0.5, sqrt(3)/2), etc.
    # Width = 2, Height = sqrt(3).
    # For hexagonal packing, adjacent hexagons in the same row are separated by 2*sqrt(3)/2 = sqrt(3) horizontally? 
    # Actually, two flat-top hexagons side by side: the rightmost point of the left is at x=1, the leftmost point of the right is at x = cx_right - 1. For them to be tangent, cx_right - 1 = 1, so cx_right = 2. But they'd overlap in the middle.
    # For non-overlapping: the distance between centers must be at least 2*sqrt(3)/2 * ... 
    # Let me just use: for two unit hexagons with the same orientation, they are disjoint iff the distance between centers >= sqrt(3) + 1? No.
    
    # Let me just try a few configurations and evaluate.
    
    best_score = float('inf')
    best_result = None
    
    # Configuration 1: Linear arrangement (baseline)
    # Configuration 2: Hexagonal cluster
    
    # Let's try a hexagonal cluster: 1 in center, 6 around, 5 more.
    # Actually, let's try: 3 rows of 4, with hexagonal offset.
    
    sqrt3 = math.sqrt(3)
    
    # Try: 4x3 grid with hexagonal offset
    # Row 0 (y=0): 4 hexagons at x = -1.5*sqrt3/2, -0.5*sqrt3/2, 0.5*sqrt3/2, 1.5*sqrt3/2
    # Wait, for flat-top hexagons, horizontal spacing for non-overlap is 2*sqrt(3)/2 = sqrt(3)
    # Vertical spacing between rows: 1.5 (since height is sqrt(3), and offset rows are shifted)
    
    # Let me just try a simple arrangement and optimize the outer.
    
    configs = []
    
    # Config A: 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * sqrt3
            y = row * 1.5
            if row % 2 == 1:
                x += sqrt3 / 2
            configs.append([x, y, 0.0])
    
    # Config B: 4 rows of 3
    configs2 = []
    for row in range(4):
        for col in range(3):
            x = (col - 1) * sqrt3
            y = (row - 1.5) * 1.5
            if row % 2 == 1:
                x += sqrt3 / 2
            configs2.append([x, y, 0.0])
    
    # Config C: Hexagonal flower (1 center + 6 ring + 5 more)
    configs3 = [[0, 0, 0.0]]
    for i in range(6):
        angle = math.radians(60 * i)
        configs3.append([sqrt3 * math.cos(angle), sqrt3 * math.sin(angle), 0.0])
    # Add 5 more around the ring
    for i in range(6):
        angle = math.radians(60 * i + 30)
        r = 2 * sqrt3
        configs3.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    configs3 = configs3[:12]
    
    # Config D: 2 rows of 6
    configs4 = []
    for row in range(2):
        for col in range(6):
            x = (col - 2.5) * sqrt3
            y = (row - 0.5) * 1.5
            configs4.append([x, y, 0.0])
    
    all_configs = [configs, configs2, configs3, configs4]
    
    for inner in all_configs:
        if len(inner) != N:
            continue
        # Check if inner hexagons are disjoint
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not _hexagons_disjoint(inner[i][0], inner[i][1], inner[i][2], 
                                          inner[j][0], inner[j][1], inner[j][2]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        
        # Find center of mass
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        
        # Try several outer angles
        for outer_angle in range(0, 60, 5):
            side = _min_outer_side(inner, cx, cy, outer_angle)
            if side < best_score:
                best_score = side
                best_result = (inner, (cx, cy), side, float(outer_angle))
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # If we have a valid result, try to optimize further
    if best_result is not None:
        inner, center, side, angle = best_result
        # Try to refine the outer angle
        for delta_angle in [x * 0.1 for x in range(-50, 51)]:
            test_angle = angle + delta_angle
            test_side = _min_outer_side(inner, center[0], center[1], test_angle)
            if test_side < side:
                side = test_side
                angle = test_angle
                best_result = (inner, center, side, angle)
        
        if time.time() - start_time > DEADLINE - 2:
            return best_result
    
    # If no valid config found, return a fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
