# EVOLVE-BLOCK-START
import numpy as np
import time


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is side length.
    """
    # Hexagonal lattice: place 12 unit hexagons in a compact cluster
    # Using a hexagonal grid with spacing that ensures non-overlap
    # Unit hexagons have width 2 (flat-to-flat) and height sqrt(3) (point-to-point)
    # For pointy-top orientation (angle=0), adjacent hexes need:
    #   horizontal spacing: 2 (side to side)
    #   vertical spacing: sqrt(3)
    #   diagonal: offset by 1 horizontally, sqrt(3)/2 vertically

    sqrt3 = np.sqrt(3.0)
    
    # Arrange 12 hexagons in a compact hexagonal cluster
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons (offset)
    # That's 12 total
    
    positions = []
    
    # Row 0 (y = 0): 3 hexagons centered
    for i in range(3):
        x = (i - 1) * 2.0
        y = 0.0
        positions.append((x, y))
    
    # Row 1 (y = sqrt(3)/2 * 2 = sqrt(3)): 4 hexagons, offset by 1
    y1 = sqrt3
    for i in range(4):
        x = (i - 1.5) * 2.0
        positions.append((x, y1))
    
    # Row 2 (y = sqrt(3) * 2): 3 hexagons centered
    y2 = 2 * sqrt3
    for i in range(3):
        x = (i - 1) * 2.0
        positions.append((x, y2))
    
    # Row 3 (y = sqrt(3) * 3): 2 hexagons, offset
    y3 = 3 * sqrt3
    for i in range(2):
        x = (i - 0.5) * 2.0
        positions.append((x, y3))
    
    # Verify we have 12
    assert len(positions) == 12, f"Expected 12, got {len(positions)}"
    
    # Compute the minimal enclosing regular hexagon
    # The outer hexagon should be centered at the centroid of all inner hexagons
    pos_array = np.array(positions)
    centroid = pos_array.mean(axis=0)
    
    # Shift positions relative to centroid
    shifted = pos_array - centroid
    
    # For a regular hexagon with side length s and angle 0 (flat-top),
    # the vertices are at distance s from center at angles 0, 60, 120, 180, 240, 300
    # A point (x, y) is inside if for each edge, the dot product with the 
    # outward normal is <= s
    
    # Actually, let's think about this differently.
    # For a flat-top regular hexagon centered at origin with side s:
    # The six vertices are at (s*cos(k*60°), s*sin(k*60°)) for k=0..5
    # The hexagon contains all points where max over 6 directions of 
    # the projection onto the direction perpendicular to each edge is <= s*sqrt(3)/2
    
    # The distance from center to each edge (apothem) is s * sqrt(3) / 2
    # For a point (x,y), the constraint for each of the 6 edges:
    # The edge normals point in directions 30°, 90°, 150°, 210°, 270°, 330°
    # (perpendicular to edges, which are at 0°, 60°, 120°, etc.)
    
    # Actually, let me just compute the required s directly.
    # For a regular hexagon with side s centered at origin, flat-top:
    # Vertex i is at (s*cos(60i°), s*sin(60i°))
    # The hexagon is the intersection of 6 half-planes.
    # Edge between vertex i and vertex i+1 has outward normal at angle 30° + 60i°
    # The distance from center to each edge is s*sqrt(3)/2
    
    # So a point (x,y) is inside iff for all k=0..5:
    # x*cos(30°+60k°) + y*sin(30°+60k°) <= s*sqrt(3)/2
    
    # We need to find the minimum s such that ALL inner hexagon vertices are inside.
    # But actually, we need the entire inner hexagons to be inside, so we need
    # all vertices of all inner hexagons to satisfy the constraint.
    
    # For each inner hexagon with center (cx, cy) and side 1, its vertices are at
    # (cx + cos(a + 60i°), cy + sin(a + 60i°)) where a is the angle.
    
    # Let's use angle 0 for all inner hexagons (flat-top)
    # Then vertices are at angles 0, 60, 120, 180, 240, 300 from center
    
    # Compute all vertices of all inner hexagons
    all_vertices = []
    for (cx, cy) in positions:
        for i in range(6):
            angle = np.radians(60 * i)
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    # Shift to centroid
    all_vertices_shifted = all_vertices - centroid
    
    # For each of the 6 edge normals of the outer hexagon, compute the max projection
    s_min = 0
    for k in range(6):
        normal_angle = np.radians(30 + 60 * k)
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        projections = all_vertices_shifted[:, 0] * nx + all_vertices_shifted[:, 1] * ny
        max_proj = np.max(projections)
        # Need s*sqrt(3)/2 >= max_proj, so s >= max_proj * 2/sqrt(3)
        s_candidate = max_proj * 2.0 / sqrt3
        s_min = max(s_min, s_candidate)
    
    # Ensure s is at least sqrt(12) as required
    s_min = max(s_min, np.sqrt(12.0))
    
    # Add a tiny margin for numerical safety
    s_min += 1e-9
    
    # Build the output arrays
    inner_hex_data = np.zeros((12, 3))
    for idx, (x, y) in enumerate(positions):
        inner_hex_data[idx, 0] = x
        inner_hex_data[idx, 1] = y
        inner_hex_data[idx, 2] = 0.0  # angle in degrees
    
    outer_hex_data = np.array([centroid[0], centroid[1], 0.0])
    
    return inner_hex_data, outer_hex_data, s_min


# EVOLVE-BLOCK-END
