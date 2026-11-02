"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

def hex_to_xy(q, r):
    """Convert axial hex coordinates to pixel coordinates."""
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def construct():
    start_time = time.time()
    
    # Generate hexagonal grid positions for N hexagons
    # Use a hexagonal cluster pattern
    positions = []
    
    # Start from center and spiral outward
    # For N=11, we need a compact arrangement
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # 11 is between 7 and 19, so we use a partial ring
    
    # Generate all hex coordinates in order of distance from center
    candidates = []
    max_ring = 3  # enough for N=11
    
    for q in range(-max_ring, max_ring + 1):
        for r in range(-max_ring, max_ring + 1):
            # Hex distance from center
            s = -q - r
            dist = max(abs(q), abs(r), abs(s))
            if dist <= max_ring:
                candidates.append((dist, q, r))
    
    # Sort by distance, then by angle for consistent ordering
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Take the first N positions (closest to center)
    selected = candidates[:N]
    
    # Convert to Cartesian coordinates
    inner = []
    for dist, q, r in selected:
        x, y = hex_to_xy(q, r)
        inner.append([x, y, 0.0])  # angle 0 for all (aligned)
    
    # Compute the minimal enclosing regular hexagon
    # The outer hexagon must contain all inner hexagons
    # Each inner hexagon has vertices at distance 1 from its center
    
    # Find the extent in all 6 directions of the outer hexagon
    # For a regular hexagon with side S centered at (cx, cy) with rotation theta,
    # the constraint is that for each inner hexagon center (xi, yi),
    # the maximum over its 6 vertices of the support function must be <= S*sqrt(3)/2
    
    # Simpler approach: find the bounding hexagon by checking all 6 normal directions
    # For a hexagon with flat sides at angles 0, 60, 120, 180, 240, 300 degrees
    
    # The inradius of the outer hexagon must be at least the maximum distance
    # from the outer center to any vertex of any inner hexagon, projected onto each normal
    
    # Let's center the outer hexagon at the centroid of the inner hexagons
    inner_arr = np.array([[p[0], p[1]] for p in inner])
    center_x = np.mean(inner_arr[:, 0])
    center_y = np.mean(inner_arr[:, 1])
    
    # For each of the 6 directions, compute the maximum extent
    # Direction angles for flat-to-flat: 0, 60, 120, 180, 240, 300
    # The inradius in direction theta is the max over all points of (p - c) dot n(theta)
    # where n(theta) is the unit normal
    
    min_side = float('inf')
    best_center = (center_x, center_y)
    best_angle = 0.0
    
    # Try different rotations of the outer hexagon
    for angle_deg in range(0, 30):  # 0 to 30 degrees, 1 degree steps
        angle_rad = np.radians(angle_deg)
        
        # 6 normal directions for the outer hexagon
        normals = []
        for k in range(6):
            theta = angle_rad + k * np.pi / 3
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        
        # For each inner hexagon, compute its 6 vertices
        # A unit hexagon centered at (cx, cy) with rotation phi has vertices at:
        # (cx + cos(phi + k*pi/3), cy + sin(phi + k*pi/3)) for k=0..5
        
        # Since all inner hexagons have angle 0, their vertices are at fixed offsets
        hex_vertices_offsets = []
        for k in range(6):
            theta = k * np.pi / 3
            hex_vertices_offsets.append(np.array([np.cos(theta), np.sin(theta)]))
        
        # For each normal direction, find the maximum support
        max_support = 0.0
        for normal in normals:
            for i in range(len(inner)):
                cx_i = inner[i][0]
                cy_i = inner[i][1]
                for voff in hex_vertices_offsets:
                    px = cx_i + voff[0]
                    py = cy_i + voff[1]
                    support = (px - center_x) * normal[0] + (py - center_y) * normal[1]
                    if support > max_support:
                        max_support = support
        
        # The inradius must be at least max_support
        # Side length S = inradius / (sqrt(3)/2) = 2 * inradius / sqrt(3)
        side = 2.0 * max_support / np.sqrt(3)
        
        if side < min_side:
            min_side = side
            best_angle = angle_deg
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Return best found so far
        pass
    
    return inner, (center_x, center_y), min_side, best_angle

# EVOLVE-BLOCK-END
