"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Place 12 unit hexagons in a hexagonal grid pattern
    # Using axial coordinates for flat-top hexagons
    # x = 1.5 * q, y = sqrt3 * (r + q/2)
    # Centers of adjacent hexagons are at distance sqrt3
    
    # Try a compact arrangement: hexagonal cluster
    # 12 hexagons: arrange in rows
    # Row pattern: 3, 4, 4, 1? Or 4, 4, 4?
    # Let's try a 3x4 block in hex grid
    
    # Axial coordinates for a compact 12-hexagon cluster
    # Using offset coordinates or axial
    # Let's use axial: (q, r) where the hex at (q,r) touches (q+1,r), (q,r+1), (q-1,r+1)
    
    # A good compact shape for 12: 
    # q=-1,r=-1; q=0,r=-1; q=1,r=-1; q=2,r=-1
    # q=-1,r=0;  q=0,r=0;  q=1,r=0;  q=2,r=0
    # q=-1,r=1;  q=0,r=1;  q=1,r=1;  q=2,r=1
    # That's 12 in a 4x3 rectangle
    
    hex_positions = []
    for q in range(-1, 3):  # 4 columns
        for r in range(-1, 2):  # 3 rows
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            hex_positions.append((x, y))
    
    # Now find minimal enclosing regular hexagon
    # A regular hexagon with center (cx, cy), side length L, rotation theta
    # contains point (x,y) if the distance from center in all 6 directions <= L
    
    # The 6 vertices of a regular hexagon with center (cx,cy), side L, rotation theta:
    # vertex_i = (cx + L*cos(theta + i*pi/3), cy + L*sin(theta + i*pi/3))
    # A point is inside if for all 6 edges, the point is on the correct side
    
    # For a regular hexagon, a point (x,y) is inside if:
    # For each of the 6 directions (normal to each edge), the projection <= apothem = L*sqrt(3)/2
    # The 6 edge normals are at angles theta + pi/6 + i*pi/3 for i=0..5
    
    # Actually, let me use a different approach: 
    # The regular hexagon with circumradius R (= side length for regular hex) 
    # A point at angle alpha from center with distance d is inside if d <= R * cos(angle_diff) / cos(pi/6) ... 
    
    # Simpler: for a regular hexagon with center at origin, side length s, rotation 0:
    # It's the intersection of 6 half-planes. The apothem (inradius) is s*sqrt(3)/2.
    # The 6 normal directions are at angles 0, 60, 120, 180, 240, 300 degrees (for flat-top)
    # A point (x,y) is inside if for each normal direction n_i, dot((x,y), n_i) <= apothem
    
    def hexagon_contains(points, cx, cy, side, angle_deg):
        """Check if all points are inside the regular hexagon."""
        apothem = side * sqrt3 / 2.0
        angle_rad = math.radians(angle_deg)
        # 6 edge normal directions (perpendicular to edges, pointing inward)
        # For a regular hexagon, edges are perpendicular to directions at angle + 30 + i*60
        for i in range(6):
            normal_angle = angle_rad + math.pi / 6.0 + i * math.pi / 3.0
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for (px, py) in points:
                dx = px - cx
                dy = py - cy
                dist = dx * nx + dy * ny
                if dist > apothem + 1e-9:
                    return False
        return True
    
    def required_side(points, cx, cy, angle_deg):
        """Find minimum side length for hexagon to contain all points."""
        angle_rad = math.radians(angle_deg)
        apothem_needed = 0.0
        for i in range(6):
            normal_angle = angle_rad + math.pi / 6.0 + i * math.pi / 3.0
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            max_proj = 0.0
            for (px, py) in points:
                dx = px - cx
                dy = py - cy
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
            if max_proj > apothem_needed:
                apothem_needed = max_proj
        return apothem_needed * 2.0 / sqrt3
    
    # Optimize: try different centers and rotations
    # Initial guess: center at centroid of points
    cx0 = sum(p[0] for p in hex_positions) / len(hex_positions)
    cy0 = sum(p[1] for p in hex_positions) / len(hex_positions)
    
    best_side = float('inf')
    best_cx, best_cy, best_angle = cx0, cy0, 0.0
    
    # Try various rotations
    for angle_deg in range(0, 60, 5):
        # Try centering at centroid
        side = required_side(hex_positions, cx0, cy0, angle_deg)
        if side < best_side:
            best_side = side
            best_cx, best_cy, best_angle = cx0, cy0, angle_deg
    
    # Refine center position
    # The optimal center for a given rotation is related to the "center" of the minimal enclosing hexagon
    # For now, let's try a few center offsets
    for dx in [-0.5, 0.0, 0.5]:
        for dy in [-0.5, 0.0, 0.5]:
            for angle_deg in range(0, 60, 5):
                side = required_side(hex_positions, cx0 + dx, cy0 + dy, angle_deg)
                if side < best_side:
                    best_side = side
                    best_cx, best_cy, best_angle = cx0 + dx, cy0 + dy, angle_deg
    
    # Check time
    if time.time() > deadline:
        # Return best found so far
        inner = [[x, y, 0.0] for (x, y) in hex_positions]
        return inner, (best_cx, best_cy), best_side, best_angle
    
    # Try a different arrangement: more hexagonal shape
    # 12 hexagons in a hexagonal pattern: 
    # Center + ring of 6 + 5 more? 
    # Or: rows of 2, 4, 4, 2? That's 12.
    hex_positions2 = []
    # Row at r=-2: 2 hexagons
    for q in [0, 1]:
        x = 1.5 * q
        y = sqrt3 * (-2 + q / 2.0)
        hex_positions2.append((x, y))
    # Row at r=-1: 4 hexagons
    for q in range(-1, 3):
        x = 1.5 * q
        y = sqrt3 * (-1 + q / 2.0)
        hex_positions2.append((x, y))
    # Row at r=0: 4 hexagons
    for q in range(-1, 3):
        x = 1.5 * q
        y = sqrt3 * (0 + q / 2.0)
        hex_positions2.append((x, y))
    # Row at r=1: 2 hexagons
    for q in [0, 1]:
        x = 1.5 * q
        y = sqrt3 * (1 + q / 2.0)
        hex_positions2.append((x, y))
    
    if len(hex_positions2) == 12:
        cx0_2 = sum(p[0] for p in hex_positions2) / len(hex_positions2)
        cy0_2 = sum(p[1] for p in hex_positions2) / len(hex_positions2)
        
        for dx in [-0.5, 0.0, 0.5]:
            for dy in [-0.5, 0.0, 0.5]:
                for angle_deg in range(0, 60, 5):
                    side = required_side(hex_positions2, cx0_2 + dx, cy0_2 + dy, angle_deg)
                    if side < best_side:
                        best_side = side
                        best_cx, best_cy, best_angle = cx0_2 + dx, cy0_2 + dy, angle_deg
                        hex_positions = hex_positions2
    
    # Return the result
    inner = [[x, y, 0.0] for (x, y) in hex_positions]
    return inner, (best_cx, best_cy), best_side, best_angle
# EVOLVE-BLOCK-END

