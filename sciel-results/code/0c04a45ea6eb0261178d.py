import math
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = math.radians(angle_deg + 60 * i)
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def support_function(verts, direction):
    """Max projection of polygon onto direction vector."""
    dx, dy = direction
    return max(x * dx + y * dy for x, y in verts)

def minimal_outer_side(inner_hexes):
    """
    Compute minimal outer hexagon side length containing all inner hexagons.
    Outer hexagon is axis-aligned (angle=0) centered at origin.
    Inner hexagons are unit side, various positions/rotations.
    
    For a regular hexagon with side s centered at origin, the support function
    in direction theta is s * max over 6 normals of projection.
    Actually, for a hexagon with vertices at angles 0,60,120,180,240,300 (flat-top),
    the support function h(theta) = s * cos(theta mod 60deg - 30deg) ... 
    Let's use the standard: a regular hexagon with side s has inradius r = s*sqrt(3)/2.
    The support function in direction (cos a, sin a) is r / cos(a - nearest_normal).
    
    Simpler: The outer hexagon must contain all inner vertices.
    For a flat-top regular hexagon centered at origin with side s:
    Vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    
    The constraint is that for each of the 6 half-plane inequalities defining the hexagon,
    all inner points must satisfy them.
    
    For flat-top hexagon with side s, the 6 half-planes are:
    x <= s
    x >= -s
    sqrt(3)*y + x <= s*sqrt(3)
    sqrt(3)*y - x <= s*sqrt(3)
    -sqrt(3)*y + x <= s*sqrt(3)
    -sqrt(3)*y - x <= s*sqrt(3)
    
    So s must be >= max over all inner points p of:
    p.x, -p.x, (sqrt(3)*p.y + p.x)/sqrt(3), (sqrt(3)*p.y - p.x)/sqrt(3), etc.
    """
    max_s = 0.0
    for (cx, cy, angle) in inner_hexes:
        verts = hex_vertices(cx, cy, 1.0, angle)
        for (px, py) in verts:
            # For flat-top hexagon centered at origin with side s:
            # The 6 constraints (half-planes):
            # 1) x <= s
            # 2) -x <= s  => x >= -s
            # 3) (sqrt(3)*y + x) / sqrt(3) <= s
            # 4) (sqrt(3)*y - x) / sqrt(3) <= s
            # 5) (-sqrt(3)*y + x) / sqrt(3) <= s
            # 6) (-sqrt(3)*y - x) / sqrt(3) <= s
            
            c1 = px
            c2 = -px
            c3 = (math.sqrt(3) * py + px) / math.sqrt(3)
            c4 = (math.sqrt(3) * py - px) / math.sqrt(3)
            c5 = (-math.sqrt(3) * py + px) / math.sqrt(3)
            c6 = (-math.sqrt(3) * py - px) / math.sqrt(3)
            
            local_max = max(c1, c2, c3, c4, c5, c6)
            if local_max > max_s:
                max_s = local_max
    return max_s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Place N hexagons in a hexagonal lattice pattern
    # For N=11, use a 3-4-3 arrangement (center row of 4, rows above/below of 3)
    # Or a compact cluster
    
    # Hexagonal lattice spacing: for unit hexagons (side=1), 
    # center-to-center distance in hex grid:
    # Horizontal: 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) for same row
    # Actually for flat-top hexagons tiled:
    # x-spacing: 3/2 * side = 1.5 for adjacent columns (offset rows)
    # y-spacing: sqrt(3) * side = sqrt(3) for adjacent rows in same column
    # For same row: x-spacing = 3 * side = 3? No.
    
    # Let me think carefully. A regular hexagon with side 1, flat-top orientation:
    # Width (left to right) = 2 * side = 2
    # Height (top to bottom) = sqrt(3) * side = sqrt(3)
    # 
    # For tiling: 
    # - Same row: centers separated by 3/2 * side = 1.5 in x? No, that's for pointy-top.
    # 
    # For flat-top hexagons:
    # - Adjacent in same row: dx = 3/2 * 2 * side? 
    # Actually, let me just use the known hex grid formula.
    # 
    # Flat-top hex, side s:
    # - Horizontal distance between centers in same row: 3s (for directly left/right neighbors in tiling)
    # Wait, no. In a hex tiling with flat-top hexagons:
    # - Each hex has 6 neighbors
    # - Two neighbors are directly left and right: distance = 3s/2 * 2? 
    # 
    # Let me just compute: a flat-top hex with side s has vertices at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # 
    # The rightmost point is at x=s, leftmost at x=-s.
    # For two hexagons side by side (sharing an edge), the right edge of left hex and left edge of right hex coincide.
    # The shared edge is vertical? No, for flat-top, the left and right edges are slanted.
    # 
    # Actually for flat-top hex, the 6 edges are:
    # Top: from (s/2, s*sqrt(3)/2) to (-s/2, s*sqrt(3)/2) -- horizontal
    # Top-right: from (s, 0) to (s/2, s*sqrt(3)/2)
    # Bottom-right: from (s, 0) to (s/2, -s*sqrt(3)/2)
    # Bottom: from (-s/2, -s*sqrt(3)/2) to (s/2, -s*sqrt(3)/2) -- horizontal
    # Bottom-left: from (-s, 0) to (-s/2, -s*sqrt(3)/2)
    # Top-left: from (-s, 0) to (-s/2, s*sqrt(3)/2)
    # 
    # So the "left" and "right" vertices are at (-s,0) and (s,0).
    # Two hexagons sharing the right edge of one with left edge of another:
    # Right edge of hex A: from (s,0) to (s/2, s*sqrt(3)/2) -- this is slanted
    # This doesn't tile simply left-right.
    # 
    # For flat-top hex tiling, neighbors are at:
    # - Directly above: (0, s*sqrt(3))
    # - Directly below: (0, -s*sqrt(3))  
    # - Upper-right: (3s/2, s*sqrt(3)/2)
    # - Lower-right: (3s/2, -s*sqrt(3)/2)
    # - Upper-left: (-3s/2, s*sqrt(3)/2)
    # - Lower-left: (-3s/2, -s*sqrt(3)/2)
    # 
    # So for s=1:
    # Same column, adjacent rows: dy = sqrt(3)
    # Adjacent columns (offset): dx = 3/2, dy = sqrt(3)/2
    
    # Let's use axial coordinates for hex grid
    # For N=11, a nice compact arrangement is a hexagon of radius 2 (which has 1+6+12=19... too many)
    # Radius 1 has 1+6=7, radius 2 has 1+6+12=19
    # For 11, let's do a 3-4-3 or similar
    
    # Simple approach: place in rows
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Total: 10... need 11
    
    # Let's do:
    # Row 0: 4
    # Row 1: 3
    # Row 2: 4
    # Total: 11
    
    # Or just a linear arrangement won't be optimal. Let's do a compact cluster.
    # 
    # For a hex grid with flat-top orientation:
    # Using offset coordinates:
    # Even rows: x = 0, 3, 6, ...
    # Odd rows: x = 1.5, 4.5, 7.5, ...
    # y = row * sqrt(3)
    
    inner_hexes = []
    
    # Try a 3-4-3 arrangement (rows of 3, 4, 3)
    # Row 0 (y=0): 4 hexagons at x = -4.5, -1.5, 1.5, 4.5
    # Row 1 (y=sqrt(3)): 3 hexagons at x = -3, 0, 3
    # Row 2 (y=2*sqrt(3)): 4 hexagons at x = -4.5, -1.5, 1.5, 4.5
    # That's 11 total
    
    sqrt3 = math.sqrt(3)
    
    # Row 0: 4 hexagons
    row0_x = [-4.5, -1.5, 1.5, 4.5]
    for x in row0_x:
        inner_hexes.append([x, 0.0, 0.0])
    
    # Row 1: 3 hexagons (offset by 1.5 in x)
    row1_x = [-3.0, 0.0, 3.0]
    for x in row1_x:
        inner_hexes.append([x, sqrt3, 0.0])
    
    # Row 2: 4 hexagons
    row2_x = [-4.5, -1.5, 1.5, 4.5]
    for x in row2_x:
        inner_hexes.append([x, 2 * sqrt3, 0.0])
    
    # Center the cluster at origin
    # Compute centroid
    cx = sum(h[0] for h in inner_hexes) / len(inner_hexes)
    cy = sum(h[1] for h in inner_hexes) / len(inner_hexes)
    for h in inner_hexes:
        h[0] -= cx
        h[1] -= cy
    
    # Compute minimal outer side
    side = minimal_outer_side(inner_hexes)
    
    # Try a few different rotations of the entire cluster to see if we can do better
    best_side = side
    best_inner = [list(h) for h in inner_hexes]
    best_outer_angle = 0.0
    
    # Try rotating the whole configuration
    for rot_deg in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline:
            break
        rotated = []
        for (x, y, a) in inner_hexes:
            # Rotate position
            rad = math.radians(rot_deg)
            nx = x * math.cos(rad) - y * math.sin(rad)
            ny = x * math.sin(rad) + y * math.cos(rad)
            rotated.append([nx, ny, a + rot_deg])
        
        s = minimal_outer_side(rotated)
        if s < best_side:
            best_side = s
            best_inner = [list(h) for h in rotated]
            best_outer_angle = 0.0  # outer stays axis-aligned
    
    # Also try rotating individual hexagons slightly
    # But first, let's try a different cluster arrangement
    
    # Alternative: more compact arrangement
    # Try a hexagonal cluster: center + 6 around + 4 more
    alt_inner = []
    # Center
    alt_inner.append([0.0, 0.0, 0.0])
    # 6 neighbors
    for i in range(6):
        angle = math.radians(60 * i)
        r = sqrt3  # distance to neighbor center
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        alt_inner.append([x, y, 0.0])
    # 4 more: add at distance 2*sqrt(3) in some directions
    for i in range(4):
        angle = math.radians(60 * i + 30)
        r = 2 * sqrt3
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        alt_inner.append([x, y, 0.0])
    
    s_alt = minimal_outer_side(alt_inner)
    if s_alt < best_side:
        best_side = s_alt
        best_inner = [list(h) for h in alt_inner]
    
    # Try yet another arrangement: 5-6-5 is too many, let's do 4-7-4? No, that's 15.
    # Let's try a more linear but staggered arrangement
    
    # For now, return the best found
    return best_inner, (0.0, 0.0), best_side, 0.0
