"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

def hex_vertex_positions(cx, cy, side, angle_deg):
    """Return 6 vertex positions of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + np.radians(60 * i)
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return vertices

def point_in_hexagon(px, py, hex_center, hex_side, hex_angle_deg):
    """Check if point (px,py) is inside the regular hexagon."""
    cx, cy = hex_center
    # Transform to hexagon's local frame
    angle_rad = np.radians(hex_angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The inradius is side * sqrt(3)/2
    # Check each of the 6 half-planes
    inradius = hex_side * np.sqrt(3) / 2
    for i in range(6):
        theta = np.radians(60 * i)
        # Normal direction
        nx = np.cos(theta)
        ny = np.sin(theta)
        # The constraint: nx*lx + ny*ly <= inradius
        if nx * lx + ny * ly > inradius + 1e-10:
            return False
    return True

def enclosing_hexagon_side(inner_positions, outer_center, outer_angle_deg):
    """Find the minimum side length of a regular hexagon (given center and angle) 
    that contains all inner hexagons."""
    cx, cy = outer_center
    # For each inner hexagon, get its vertices
    # The outer hexagon must contain all these vertices
    # For a regular hexagon with side L, center (cx,cy), angle a:
    # A point (px,py) is inside if for all 6 directions, the projection <= L*sqrt(3)/2
    
    # Transform all points to local frame of outer hexagon
    angle_rad = np.radians(outer_angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    
    max_proj = 0.0
    for (ix, iy, iangle) in inner_positions:
        verts = hex_vertex_positions(ix, iy, 1.0, iangle)
        for (vx, vy) in verts:
            dx = vx - cx
            dy = vy - cy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for j in range(6):
                theta = np.radians(60 * j)
                nx = np.cos(theta)
                ny = np.sin(theta)
                proj = nx * lx + ny * ly
                if proj > max_proj:
                    max_proj = proj
    
    # inradius = L * sqrt(3)/2, so L = max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
    return 2.0 * max_proj / np.sqrt(3)

def try_configuration(inner_positions):
    """Try different outer hexagon centers and angles, return best side length."""
    # Try outer center at centroid of inner hexagons
    xs = [p[0] for p in inner_positions]
    ys = [p[1] for p in inner_positions]
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    best_side = float('inf')
    # Try various angles for the outer hexagon
    for angle in range(0, 360, 15):
        side = enclosing_hexagon_side(inner_positions, (cx, cy), angle)
        if side < best_side:
            best_side = side
    
    # Also try fine-tuning center
    # The optimal center might not be exactly the centroid
    # Try a small grid around the centroid
    for dcx in np.linspace(-0.5, 0.5, 11):
        for dcy in np.linspace(-0.5, 0.5, 11):
            for angle in range(0, 360, 30):
                side = enclosing_hexagon_side(inner_positions, (cx + dcx, cy + dcy), angle)
                if side < best_side:
                    best_side = side
    
    return best_side

def generate_honeycomb_clusters(n):
    """Generate compact clusters of n hexagons on a honeycomb lattice."""
    # Honeycomb lattice: centers at i*(sqrt(3), 0) + j*(sqrt(3)/2, 3/2)
    # Let's use axial coordinates (q, r) -> cartesian
    # For flat-top hexagons with side 1:
    # q-axis: (sqrt(3), 0)
    # r-axis: (sqrt(3)/2, 3/2)
    
    sqrt3 = np.sqrt(3)
    ax1 = np.array([sqrt3, 0.0])
    ax2 = np.array([sqrt3/2, 3.0/2])
    
    # Generate candidate positions in axial coordinates
    # We want compact clusters, so use small q,r values
    max_coord = 3
    candidates = []
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            pos = q * ax1 + r * ax2
            candidates.append((q, r, pos[0], pos[1]))
    
    # Try all subsets of size n? That's too many.
    # Instead, try specific patterns.
    
    patterns = []
    
    # Pattern 1: Linear arrangement
    for direction in [ax1, ax2, ax1 - ax2, -ax1, -ax2, ax2 - ax1]:
        pts = []
        for i in range(n):
            p = i * direction
            pts.append((p[0], p[1], 0.0))
        patterns.append(pts)
    
    # Pattern 2: Hexagonal cluster (center + rings)
    # Center at origin
    axial_positions = [(0, 0)]
    # Ring 1: 6 positions
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    axial_positions.extend(ring1)
    # Ring 2: 12 positions
    ring2 = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), 
             (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)]
    axial_positions.extend(ring2)
    
    # Try subsets that include the center for compactness
    # For n=11, try center + all of ring1 + 4 from ring2
    if n >= 7:
        base = axial_positions[:7]  # center + ring1
        remaining_needed = n - 7
        if remaining_needed > 0 and remaining_needed <= len(ring2):
            # Try all combinations of 4 from ring2
            from itertools import combinations
            for combo in combinations(range(len(ring2)), remaining_needed):
                pts_axial = base + [ring2[i] for i in combo]
                pts = []
                for (q, r) in pts_axial:
                    pos = q * ax1 + r * ax2
                    pts.append((pos[0], pos[1], 0.0))
                patterns.append(pts)
    
    # Pattern 3: 2-row arrangement
    # Row 1: ceil(n/2) hexagons, Row 2: floor(n/2) hexagons, offset
    row1_count = (n + 1) // 2
    row2_count = n // 2
    pts = []
    for i in range(row1_count):
        pos = i * ax1
        pts.append((pos[0], pos[1], 0.0))
    for i in range(row2_count):
        pos = (i + 0.5) * ax1 + ax2
        pts.append((pos[0], pos[1], 0.0))
    patterns.append(pts)
    
    # Pattern 4: 3-row arrangement
    r1 = n // 3
    r2 = (n + 2) // 3
    r3 = n - r1 - r2
    if r1 >= 0 and r2 >= 0 and r3 >= 0:
        pts = []
        for i in range(r1):
            pos = i * ax1
            pts.append((pos[0], pos[1], 0.0))
        for i in range(r2):
            pos = (i + 0.5) * ax1 + ax2
            pts.append((pos[0], pos[1], 0.0))
        for i in range(r3):
            pos = i * ax1 + 2 * ax2
            pts.append((pos[0], pos[1], 0.0))
        patterns.append(pts)
    
    return patterns

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = float('inf')
    best_angle = 0.0
    
    patterns = generate_honeycomb_clusters(N)
    
    for pattern in patterns:
        if time.time() > deadline - 10.0:
            break
        
        side = try_configuration(pattern)
        if side < best_side:
            best_side = side
            best_inner = pattern
            # Find optimal center and angle
            xs = [p[0] for p in pattern]
            ys = [p[1] for p in pattern]
            cx = np.mean(xs)
            cy = np.mean(ys)
            
            best_angle_for_this = 0.0
            best_side_for_this = float('inf')
            for angle in range(0, 360, 5):
                s = enclosing_hexagon_side(pattern, (cx, cy), angle)
                if s < best_side_for_this:
                    best_side_for_this = s
                    best_angle_for_this = angle
            
            best_center = (cx, cy)
            best_angle = best_angle_for_this
            best_side = best_side_for_this
    
    # If no good pattern found, fallback to seed
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
