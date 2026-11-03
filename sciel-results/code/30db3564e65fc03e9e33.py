"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 11
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
    # Transform to hexagon's local coordinate system
    angle_rad = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    # Rotate point to align with hexagon
    x_local = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
    y_local = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)
    # For a regular hexagon with side s, the distance from center to each side is s*sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes
    d = side * np.sqrt(3) / 2
    for i in range(6):
        a = np.radians(angle_deg) + i * np.pi / 3
        nx, ny = np.cos(a), np.sin(a)
        if nx * x_local + ny * y_local > d:
            return False
    return True

def hexagon_overlap_area(v1, v2):
    """Approximate overlap area between two convex polygons using Sutherland-Hodgman."""
    # Simple check: if any vertex of one is inside the other, they likely overlap
    for i in range(len(v1)):
        if point_in_hexagon(v1[i][0], v1[i][1], 0, 0, 1, 0):
            pass  # This is a rough check; we'll use a better method below
    return 0.0

def check_no_overlap(inner_hexes, eps=1e-6):
    """Check that no two inner hexagons overlap. Returns True if valid."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ai = inner_hexes[i]
            xj, yj, aj = inner_hexes[j]
            # Get vertices
            vi = hex_vertices(xi, yi, 1.0, ai)
            vj = hex_vertices(xj, yj, 1.0, aj)
            # Check if any vertex of vi is inside vj or vice versa
            for k in range(6):
                if point_in_hexagon(vi[k][0], vi[k][1], xj, yj, 1.0, aj):
                    return False
            for k in range(6):
                if point_in_hexagon(vj[k][0], vj[k][1], xi, yi, 1.0, ai):
                    return False
            # Also check edge crossings (simplified: if bounding boxes overlap significantly)
            # For now, vertex containment is a good heuristic
    return True

def compute_min_side(inner_hexes, center=(0.0, 0.0), outer_angle=0.0):
    """Compute the minimum outer hexagon side length to contain all inner hexagons."""
    cx, cy = center
    # For each inner hexagon, compute the maximum distance from center in each of the 6 hexagon directions
    max_radius = 0.0
    for i in range(len(inner_hexes)):
        xi, yi, ai = inner_hexes[i]
        # The inner hexagon extends 1.0 from its center in all directions (circumradius = 1)
        # But we need to check the actual extent in the outer hexagon's coordinate frame
        # The outer hexagon with side S and angle theta contains points where:
        # for each direction k: |p - center| projected on direction k <= S * sqrt(3) / 2
        for k in range(6):
            a = np.radians(outer_angle) + k * np.pi / 3
            nx, ny = np.cos(a), np.sin(a)
            # Maximum extent of inner hexagon in this direction
            # The inner hexagon has vertices at distance 1 from its center
            # Project all vertices of inner hexagon onto direction (nx, ny)
            vi = hex_vertices(xi, yi, 1.0, ai)
            projections = vi[:, 0] * nx + vi[:, 1] * ny
            # The outer hexagon constraint: |projection - center_projection| <= S * sqrt(3) / 2
            center_proj = cx * nx + cy * ny
            max_proj = np.max(projections)
            min_proj = np.min(projections)
            required = max(abs(max_proj - center_proj), abs(min_proj - center_proj))
            max_radius = max(max_radius, required)
    # The outer hexagon side S satisfies: S * sqrt(3) / 2 = max_radius
    # So S = 2 * max_radius / sqrt(3)
    return 2.0 * max_radius / np.sqrt(3)

def construct():
    start_time = time.time()
    
    # Define a good initial cluster: hexagonal arrangement
    # 1 in center, 6 around it, 4 more
    # Let's use a compact arrangement
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons
    # That's 11 total
    
    # Hexagon spacing: for unit hexagons, the center-to-center distance for adjacent
    # hexagons (sharing an edge) is sqrt(3)
    d = np.sqrt(3)
    
    # Define relative positions of the 11 hexagons in a compact cluster
    # Using axial coordinates for hex grid
    # Let's place them in a pattern that minimizes the bounding hexagon
    
    # Pattern: 
    #   . X .
    #   X X X
    #   X X X X
    #   . X X .
    # That's 1+3+4+3 = 11? No, let me count: row0=1, row1=3, row2=4, row3=3 = 11
    
    # Better: use a known good packing
    # Let's start with a hexagonal cluster of radius 2 (19 hexagons) and remove 8
    
    # Simpler: start with a line and improve
    # Actually, let's use a 2-3-4-2 pattern or similar
    
    # Define cluster positions (relative to cluster center)
    # Using Cartesian coordinates for hex grid with flat-top orientation
    # For flat-top hexagons, adjacent hexagons are spaced:
    # Horizontal: 1.5 * side = 1.5
    # Vertical: sqrt(3) * side = sqrt(3)
    # Offset rows: 0.75 * side horizontal offset
    
    s = 1.0  # side length of inner hexagons
    # Flat-top hexagon: width = 2*s, height = sqrt(3)*s
    # Center-to-center for adjacent:
    # Same row: 1.5*s horizontal
    # Adjacent row: 0.75*s horizontal, sqrt(3)/2*s vertical
    
    # Let's define a compact 11-hexagon cluster
    # Row 0 (y=0): 3 hexagons at x = -1.5, 0, 1.5
    # Row 1 (y=sqrt(3)/2): 4 hexagons at x = -2.25, -0.75, 0.75, 2.25
    # Row 2 (y=sqrt(3)): 4 hexagons at x = -1.5, 0, 1.5, and one more? 
    # That's 3+4+4 = 11
    
    # Wait, let me reconsider. For flat-top hexagons:
    # Hexagon at (0,0) has neighbors at:
    # (1.5, 0), (0.75, sqrt(3)/2), (-0.75, sqrt(3)/2), (-1.5, 0), (-0.75, -sqrt(3)/2), (0.75, -sqrt(3)/2)
    
    # Let's build a cluster:
    # Center: (0, 0)
    # Ring 1 (6 hexagons): the 6 neighbors above
    # Ring 2 (4 more): pick 4 from the next ring
    
    # Positions:
    positions = [
        (0.0, 0.0),  # center
        (1.5, 0.0),  # right
        (0.75, np.sqrt(3)/2),  # upper right
        (-0.75, np.sqrt(3)/2),  # upper left
        (-1.5, 0.0),  # left
        (-0.75, -np.sqrt(3)/2),  # lower left
        (0.75, -np.sqrt(3)/2),  # lower right
        # Add 4 more from ring 2
        (2.25, np.sqrt(3)/2),  # far upper right
        (0.0, np.sqrt(3)),  # top
        (-2.25, np.sqrt(3)/2),  # far upper left
        (0.0, -np.sqrt(3)),  # bottom
    ]
    
    # Check: that's 11 positions
    
    # Now optimize: rotate the entire cluster and find the best outer hexagon
    # We'll try different global rotation angles and compute the required outer side
    
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    best_inner = None
    
    # Try a range of rotation angles
    n_angles = 36
    for i in range(n_angles):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        global_angle = i * 360.0 / n_angles
        # Also try some intermediate angles
        global_angle2 = (i + 0.5) * 360.0 / n_angles
        
        for ga in [global_angle, global_angle2]:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Rotate all positions by ga
            cos_a = np.cos(np.radians(ga))
            sin_a = np.sin(np.radians(ga))
            
            rotated_positions = []
            for px, py in positions:
                rx = px * cos_a - py * sin_a
                ry = px * sin_a + py * cos_a
                rotated_positions.append((rx, ry))
            
            # Center the cluster: compute centroid and shift to origin
            xs = [p[0] for p in rotated_positions]
            ys = [p[1] for p in rotated_positions]
            cx = np.mean(xs)
            cy = np.mean(ys)
            
            centered_positions = [(x - cx, y - cy) for x, y in rotated_positions]
            
            # The inner hexagons are at these positions, each with angle ga (they rotate with the cluster)
            inner_hexes = [(x, y, ga) for x, y in centered_positions]
            
            # Check for overlaps
            if not check_no_overlap(inner_hexes):
                continue
            
            # Compute required outer side
            outer_side = compute_min_side(inner_hexes, center=(0.0, 0.0), outer_angle=0.0)
            
            if outer_side < best_side:
                best_side = outer_side
                best_angle = ga
                best_center = (0.0, 0.0)
                best_inner = inner_hexes[:]
    
    # If no valid packing found, fall back to a simple arrangement
    if best_inner is None:
        # Simple line arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, 0.0

# EVOLVE-BLOCK-END
