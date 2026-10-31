"""Rewritten program for packing unit regular hexagons inside a regular hexagon."""
import numpy as np
from scipy.spatial import cKDTree

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Pack unit hexagons in a honeycomb-like arrangement inside a larger hexagon.
    # We'll try to fit them in a hexagonal grid pattern, then compute the minimal bounding hexagon.
    
    # For N hexagons, we can arrange them in a hexagonal cluster.
    # A hexagonal cluster of radius r contains 1 + 3r(r+1) hexagons.
    # For r=1: 7, r=2: 19, r=3: 37...
    # N=11 doesn't fit a perfect hexagonal cluster.
    
    # Let's use a more systematic approach:
    # Place hexagons in rows, with each row offset by sqrt(3)/2 in y and 1.5 in x (for flat-top orientation)
    # Actually, for pointy-top hexagons with side 1:
    #   Width = sqrt(3), Height = 2
    #   Horizontal spacing = sqrt(3), Vertical spacing = 1.5
    #   Odd rows offset by sqrt(3)/2
    
    # Let's use flat-top hexagons:
    #   Width = 2, Height = sqrt(3)
    #   Horizontal spacing = 1.5, Vertical spacing = sqrt(3)
    #   Odd columns offset by sqrt(3)/2
    
    # I'll try a simple grid arrangement and then compute the bounding hexagon.
    
    # For N=11, let's try a 3-4-3 or 4-3-4 arrangement
    # Or just use a linear/hexagonal packing
    
    # Let's place them in a hexagonal lattice pattern
    positions = []
    angle = 0.0  # All same orientation for simplicity
    
    # Hexagonal lattice: for flat-top hexagons
    # dx = 1.5 (horizontal distance between centers in same row)
    # dy = sqrt(3) (vertical distance between rows)
    # Odd rows shifted by 0.75
    
    sqrt3 = np.sqrt(3)
    
    # Try to arrange in a compact hexagonal shape
    # For N=11, use rows of 3, 4, 3, 1? Or 2, 3, 4, 2?
    # Let's try: row0: 3, row1: 4, row2: 3, row3: 1 -> total 11
    # Or more symmetric: 2, 3, 4, 2 = 11
    
    rows = [2, 3, 4, 2]
    # Total = 11
    
    y_pos = 0.0
    row_y = []
    for i, count in enumerate(rows):
        row_y.append(y_pos)
        y_pos += sqrt3  # vertical spacing
    
    # Center the arrangement
    total_height = (len(rows) - 1) * sqrt3
    y_offset = -total_height / 2
    
    for i, count in enumerate(rows):
        y = row_y[i] + y_offset
        # Horizontal centering for this row
        # Max width in row is 4 hexagons: width = 3 * 1.5 = 4.5
        # So offset for row of 'count' hexagons:
        # Center of row should be at x=0
        # First hex center at x = -(count-1)*1.5/2
        for j in range(count):
            x = (j - (count - 1) / 2.0) * 1.5
            positions.append([x, y, angle])
    
    # Now compute the bounding regular hexagon
    # The bounding hexagon needs to contain all these unit hexagons.
    # Each unit hexagon has circumradius 1 (distance from center to vertex).
    # So we need a hexagon that contains all points at distance 1 from each center.
    
    # For a regular hexagon with side length S and center (cx, cy) and rotation theta:
    # A point (px, py) is inside if for all 6 directions, the projection is <= S * cos(30°) = S * sqrt(3)/2
    # Actually, the apothem of the outer hexagon is S * sqrt(3)/2.
    # The distance from center to any side must be >= distance from (px,py) to that side + 1 (circumradius of inner hex)
    
    # Let's compute the minimal S by checking constraints.
    # For a hexagon centered at origin with rotation 0 (flat-top), the 6 outward normals are at angles 0, 60, 120, 180, 240, 300.
    # The constraint for each direction i: dot((px-cx, py-cy), n_i) + 1 <= S * sqrt(3)/2
    
    # We want to minimize S. Let's first try center at (0,0) and rotation 0.
    
    # Compute required S for center (0,0), rotation 0
    max_proj = 0
    for px, py, _ in positions:
        for k in range(6):
            theta = k * np.pi / 3
            nx = np.cos(theta)
            ny = np.sin(theta)
            proj = px * nx + py * ny
            max_proj = max(max_proj, proj)
    
    S = (max_proj + 1.0) / (sqrt3 / 2)
    
    # Let's also try different centers and rotations to see if we can do better
    best_S = S
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try a few rotations
    for rot_deg in [0, 30, 60, 90, 120, 150]:
        rot = np.radians(rot_deg)
        max_proj = 0
        for px, py, _ in positions:
            # Rotate point by -rot to align with hexagon axes
            xr = px * np.cos(rot) + py * np.sin(rot)
            yr = -px * np.sin(rot) + py * np.cos(rot)
            for k in range(6):
                theta = k * np.pi / 3
                nx = np.cos(theta)
                ny = np.sin(theta)
                proj = xr * nx + yr * ny
                max_proj = max(max_proj, proj)
        S_candidate = (max_proj + 1.0) / (sqrt3 / 2)
        if S_candidate < best_S:
            best_S = S_candidate
            best_angle = rot_deg
    
    # Try shifting the center
    # The optimal center should be near the centroid of the points
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    
    for rot_deg in [0, 30, 60, 90, 120, 150]:
        rot = np.radians(rot_deg)
        max_proj = 0
        for px, py, _ in positions:
            dx = px - cx
            dy = py - cy
            xr = dx * np.cos(rot) + dy * np.sin(rot)
            yr = -dx * np.sin(rot) + dy * np.cos(rot)
            for k in range(6):
                theta = k * np.pi / 3
                nx = np.cos(theta)
                ny = np.sin(theta)
                proj = xr * nx + yr * ny
                max_proj = max(max_proj, proj)
        S_candidate = (max_proj + 1.0) / (sqrt3 / 2)
        if S_candidate < best_S:
            best_S = S_candidate
            best_center = (cx, cy)
            best_angle = rot_deg
    
    inner = positions
    return inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
