"""Packing unit regular hexagons inside a regular hexagon using grid-based Minkowski difference approach."""
import numpy as np
from scipy.spatial import cKDTree
import math
import random

N = 11

def hexagon_vertices(cx, cy, side, angle_rad):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    verts = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append((x, y))
    return verts

def point_in_hexagon(px, py, cx, cy, side, angle_rad):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side length and rotation."""
    # Transform point to hexagon local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # Regular hexagon with side 'side' has inradius = side * sqrt(3)/2
    inradius = side * math.sqrt(3) / 2
    circumradius = side
    
    # In local frame, hexagon vertices are at angles 0, 60, 120, 180, 240, 300
    # The hexagon can be described by: for each of 6 edges, the point must be on the correct side
    # Edge i connects vertex i to vertex i+1
    # Normal to edge i points outward at angle (i+0.5)*60 degrees
    
    for i in range(6):
        edge_angle = (i + 0.5) * math.pi / 3
        nx = math.cos(edge_angle)
        ny = math.sin(edge_angle)
        # Distance from center along normal direction
        dist = lx * nx + ly * ny
        # The support function of a regular hexagon with circumradius R at angle theta
        # For a regular hexagon with circumradius R, the width in direction theta is:
        # If theta is near a vertex direction, width = R
        # If theta is near an edge normal direction, width = R * sqrt(3)/2
        # More precisely, for angle theta from center:
        # The boundary is at distance R * cos(pi/6) / cos((theta mod pi/3) - pi/6) ... 
        # Actually let's use the half-width in direction (nx, ny)
        # For regular hexagon with circumradius R, the half-width in direction making angle phi with nearest vertex is:
        # h(phi) = R * cos(pi/6) / cos(phi - pi/6) for phi in [0, pi/3]
        # But simpler: the hexagon is intersection of 6 half-planes.
        # Edge i is perpendicular to direction at angle (i+0.5)*pi/3
        # The distance from center to edge i is inradius = R*sqrt(3)/2
        # So point is inside if lx*nx + ly*ny <= inradius for all 6 edges
        if dist > inradius + 1e-9:
            return False
    return True

def hexagon_distance(p1, p2, r1, r2, a1, a2):
    """Approximate minimum distance between two hexagons. Returns negative if overlapping."""
    # Simple check: distance between centers vs sum of circumradii
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    d = math.sqrt(dx*dx + dy*dy)
    # This is a rough approximation; for better accuracy we'd need more work
    # For now, use circumradius sum as upper bound on non-overlap distance
    # But this is conservative (may allow overlaps)
    # Better: check if one hexagon's vertices are inside the other
    # For efficiency in grid approach, we'll use a simpler criterion
    
    # The Minkowski sum of two unit hexagons (with optimal relative rotation) 
    # has a certain shape. For same orientation, the Minkowski sum is a hexagon with side 2.
    # The "forbidden zone" for center of hex2 given hex1 is the Minkowski sum of hex1 and -hex2.
    # For identical hexagons with same orientation, this is a hexagon with side 2*side.
    # The inradius of this forbidden hexagon is 2*side*sqrt(3)/2 = side*sqrt(3)
    # So centers must be at least side*sqrt(3) apart for non-overlap (same orientation)
    
    # For different orientations, it's more complex. Let's use a conservative estimate.
    # The maximum extent of a unit hexagon is its circumradius = 1.
    # So if center distance > 2, they definitely don't overlap.
    # If center distance < sqrt(3) ~ 1.732, they definitely overlap (for same orientation).
    # For different orientations, the minimum distance for non-overlap is somewhere in between.
    
    # Let's do a more careful check using the support function approach
    # Sample several directions to estimate if hexagons overlap
    
    # For efficiency, use a quick check first
    if d > 2.0:
        return d - 2.0  # Definitely non-overlapping
    
    # Check vertices of hex1 against hex2 and vice versa
    verts1 = hexagon_vertices(p1[0], p1[1], r1, a1)
    verts2 = hexagon_vertices(p2[0], p2[1], r2, a2)
    
    # Check if any vertex of hex1 is inside hex2
    for v in verts1:
        if point_in_hexagon(v[0], v[1], p2[0], p2[1], r2, a2):
            return -1.0  # Overlapping
    
    # Check if any vertex of hex2 is inside hex1
    for v in verts2:
        if point_in_hexagon(v[0], v[1], p1[0], p1[1], r1, a1):
            return -1.0  # Overlapping
    
    # Check edge-edge intersections would be more thorough but expensive
    # For now, if no vertices are inside, assume non-overlapping
    # This is not perfectly accurate but good enough for grid-based approach
    return d - 2.0  # Approximate clearance

def construct():
    """
    Construct packing of N unit hexagons inside a regular hexagon.
    Returns: (inner, center, side, angle)
    inner: list of [x, y, angle_degrees] for each inner hexagon
    center: (x, y) of outer hexagon
    side: side length of outer hexagon
    angle: rotation of outer hexagon in degrees
    """
    # Strategy: Use a grid-based approach to find positions
    # The outer hexagon side length determines the container
    # We want to minimize the outer side length
    
    # For N hexagons, a reasonable starting estimate:
    # If we arrange them in a hexagonal pattern, the container needs to be larger
    # A single hexagon needs side >= 1 (actually slightly more due to rotation)
    # For N hexagons in a line, side ~ N * sqrt(3)/2 + 1
    
    # Let's try different arrangements and find the minimum side length
    
    best_result = None
    best_side = float('inf')
    
    # Try a hexagonal close-packed arrangement
    # Arrange N hexagons in concentric rings or a hexagonal grid
    
    # First, let's try a simple hexagonal grid arrangement
    # In a hexagonal grid, adjacent hexagons (same orientation) have center distance sqrt(3)
    
    # Generate positions in a hexagonal grid pattern
    def hex_grid_positions(n):
        """Generate n positions in a hexagonal grid pattern (same orientation)."""
        positions = []
        # Start from center and expand
        if n <= 1:
            return [(0.0, 0.0)]
        
        # Hexagonal numbers: 1, 7, 19, 37, 61, ...
        # For a hexagonal arrangement with radius r, we have 3r(r+1)+1 hexagons
        
        # Find how many rings we need
        rings_needed = 0
        total = 1
        while total < n:
            rings_needed += 1
            total = 3 * rings_needed * (rings_needed + 1) + 1
        
        # Generate all positions up to rings_needed
        all_pos = [(0.0, 0.0)]
        for r in range(1, rings_needed + 1):
            # 6r positions in ring r
            for i in range(6 * r):
                # Position on hexagonal grid
                # Ring r has 6r cells
                # Direction changes every r cells
                seg = i // r
                pos_in_seg = i % r
                # Direction angle for segment seg
                angle = seg * math.pi / 3
                # Starting point of this segment
                if seg == 0:
                    sx = r * math.sqrt(3)
                    sy = 0
                elif seg == 1:
                    sx = r * math.sqrt(3) / 2
                    sy = r * 1.5
                elif seg == 2:
                    sx = -r * math.sqrt(3) / 2
                    sy = r * 1.5
                elif seg == 3:
                    sx = -r * math.sqrt(3)
                    sy = 0
                elif seg == 4:
                    sx = -r * math.sqrt(3) / 2
                    sy = -r * 1.5
                else:
                    sx = r * math.sqrt(3) / 2
                    sy = -r * 1.5
                
                # Step along the segment
                # The direction of movement within a segment
                if seg == 0:
                    dx, dy = 0, 1.5
                elif seg == 1:
                    dx, dy = -math.sqrt(3)/2, 1.5
                elif seg == 2:
                    dx, dy = -math.sqrt(3), 0
                elif seg == 3:
                    dx, dy = 0, -1.5
                elif seg == 4:
                    dx, dy = math.sqrt(3)/2, -1.5
                else:
                    dx, dy = math.sqrt(3), 0
                
                x = sx + pos_in_seg * dx
                y = sy + pos_in_seg * dy
                all_pos.append((x, y))
        
        # Take first n positions
        return all_pos[:n]
    
    # Try hexagonal grid arrangement
    positions = hex_grid_positions(N)
    
    # All hexagons have same orientation (0 degrees)
    angles = [0.0] * N
    
    # Find the minimum enclosing hexagon
    # The outer hexagon must contain all inner hexagons
    # For a regular hexagon with side S and rotation 0, centered at origin:
    # A point (x,y) is inside if for all 6 edge normals, the projection is <= S*sqrt(3)/2
    
    def min_enclosing_hexagon_side(positions, angles, outer_angle_rad=0.0):
        """Find minimum side length of regular hexagon (with given rotation) that contains all hexagons."""
        # For each inner hexagon, compute the maximum extent in each of the 6 edge normal directions
        # The outer hexagon's half-width in direction i must be >= max over all inner hexagons of their extent in that direction
        
        max_extents = [0.0] * 6
        
        for (px, py), ang_deg in zip(positions, angles):
            ang_rad = math.radians(ang_deg)
            # For this inner hexagon, compute its support function in each of the 6 outer hexagon edge normal directions
            for i in range(6):
                # Edge normal direction for outer hexagon
                edge_angle = outer_angle_rad + (i + 0.5) * math.pi / 3
                nx = math.cos(edge_angle)
                ny = math.sin(edge_angle)
                
                # The extent of the inner hexagon in direction (nx, ny) from the outer center
                # = projection of center + support function of inner hexagon in that direction
                center_proj = px * nx + py * ny
                
                # Support function of a regular hexagon with side 1, rotation ang_rad, in direction (nx, ny)
                # The support function h(theta) of a regular hexagon with circumradius R=1:
                # h(theta) = cos(pi/6) / cos((theta - ang_rad) mod (pi/3) - pi/6) * ... 
                # Actually, for a regular hexagon with circumradius R, the support function is:
                # h(phi) = R * cos(pi/6) / cos((phi mod pi/3) - pi/6) where phi is angle relative to hexagon's vertex directions
                # Let me compute this properly
                
                # The hexagon has vertices at angles ang_rad + k*pi/3 for k=0..5
                # The support function in direction theta is max over vertices of v_k . (cos(theta), sin(theta))
                # But for a convex polygon, h(theta) = max_k (v_k . u) where u = (cos(theta), sin(theta))
                # For a regular hexagon centered at origin with circumradius R:
                # h(theta) = R * max_k cos(theta - ang_rad - k*pi/3)
                # = R * cos(min_k |theta - ang_rad - k*pi/3| mod pi) ... 
                # Actually, since cos is periodic and the hexagon has 6-fold symmetry:
                # h(theta) = R * cos(d) where d = distance from theta to nearest vertex direction, but capped
                # For a regular hexagon, h(theta) = R * cos(pi/6) / cos((theta mod pi/3) - pi/6) is not quite right
                # Let's just compute it directly: h(theta) = max_k cos(theta - ang_rad - k*pi/3) * R
                
                theta = math.atan2(ny, nx)
                max_cos = 0.0
                for k in range(6):
                    c = math.cos(theta - ang_rad - k * math.pi / 3)
                    if c > max_cos:
                        max_cos = c
                support = max_cos  # R=1
                
                extent = center_proj + support
                if extent > max_extents[i]:
                    max_extents[i] = extent
        
        # The outer hexagon with side S has half-width S*sqrt(3)/2 in each edge normal direction
        # So we need S*sqrt(3)/2 >= max_extents[i] for all i
        # S >= max(max_extents) * 2/sqrt(3)
        max_ext = max(max_extents)
        side = max_ext * 2.0 / math.sqrt(3)
        return side
    
    # Compute side length for hexagonal grid arrangement
    side = min_enclosing_hexagon_side(positions, angles, 0.0)
    
    inner = [[px, py, ang] for (px, py), ang in zip(positions, angles)]
    center = (0.0, 0.0)
    angle_deg = 0.0
    
    # Try rotating the outer hexagon to see if we can get a smaller side
    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
        outer_angle_rad = math.radians(outer_angle_deg)
        s = min_enclosing_hexagon_side(positions, angles, outer_angle_rad)
        if s < side:
            side = s
            angle_deg = outer_angle_deg
    
    # Also try with outer hexagon rotated by 30 degrees relative to inner
    # And try centering differently
    
    # Try a linear arrangement as well
    linear_positions = [(k * math.sqrt(3), 0.0) for k in range(N)]
    linear_angles = [0.0] * N
    linear_side = min_enclosing_hexagon_side(linear_positions, linear_angles, 0.0)
    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
        outer_angle_rad = math.radians(outer_angle_deg)
        s = min_enclosing_hexagon_side(linear_positions, linear_angles, outer_angle_rad)
        if s < linear_side:
            linear_side = s
    
    if linear_side < side:
        positions = linear_positions
        angles = linear_angles
        side = linear_side
        inner = [[px, py, ang] for (px, py), ang in zip(positions, angles)]
    
    # Try a 2D grid arrangement
    # Arrange in rows
    rows = []
    remaining = N
    row_width = int(math.ceil(math.sqrt(N * 2 / math.sqrt(3))))  # Rough estimate
    # Actually, let's try different row configurations
    for num_rows in range(1, N + 1):
        # Distribute N items into num_rows rows
        base = N // num_rows
        extra = N % num_rows
        row_sizes = [base + (1 if i < extra else 0) for i in range(num_rows)]
        
        pos_list = []
        ang_list = []
        y_offset = 0
        for row_size in row_sizes:
            x_start = -(row_size - 1) * math.sqrt(3) / 2
            for j in range(row_size):
                x = x_start + j * math.sqrt(3)
                pos_list.append((x, y_offset))
                ang_list.append(0.0)
            y_offset += 1.5  # Vertical spacing for hexagonal packing
        
        if len(pos_list) == N:
            s = min_enclosing_hexagon_side(pos_list, ang_list, 0.0)
            for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
                outer_angle_rad = math.radians(outer_angle_deg)
                s2 = min_enclosing_hexagon_side(pos_list, ang_list, outer_angle_rad)
                if s2 < s:
                    s = s2
            if s < side:
                side = s
                positions = pos_list
                angles = ang_list
                inner = [[px, py, ang] for (px, py), ang in zip(positions, angles)]
    
    return inner, center, side, angle_deg

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
