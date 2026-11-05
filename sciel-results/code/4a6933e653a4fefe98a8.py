"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    apothem_inner = sqrt3 / 2
    
    # Generate hexagonal grid positions in axial coordinates
    # Convert to Cartesian for unit hexagons with flat-top orientation
    positions = []
    
    # Center
    positions.append((0.0, 0.0))
    
    # Add rings until we have at least N positions
    for ring in range(1, 20):
        # 6 * ring positions in this ring
        # The 6 starting points and directions for each side of the hexagon ring
        start_points = [
            (0, ring),    # top
            (ring, -ring), # top-right
            (ring, 0),     # right
            (0, -ring),    # bottom
            (-ring, ring), # bottom-left
            (-ring, 0),    # left
        ]
        directions = [
            (1, 0),   # top side: go right
            (1, -1),  # top-right side: go down-right
            (0, -1),  # right side: go down
            (-1, 0),  # bottom side: go left
            (-1, 1),  # bottom-left side: go up-left
            (0, 1),   # left side: go up
        ]
        
        for side in range(6):
            sq, sr = start_points[side]
            dq, dr = directions[side]
            for pos in range(ring):
                q = sq + pos * dq
                r = sr + pos * dr
                # Axial to Cartesian for flat-top hexagons
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                positions.append((x, y))
        
        if len(positions) >= N:
            break
    
    # Take only N positions
    positions = positions[:N]
    
    # Compute minimum outer hexagon side length
    # For aligned hexagons, check all 6 edge normal directions
    angles = [i * np.pi / 3 for i in range(6)]
    
    max_ratio = 0.0
    for (x, y) in positions:
        for theta in angles:
            extent = x * np.cos(theta) + y * np.sin(theta) + apothem_inner
            ratio = extent / apothem_inner
            if ratio > max_ratio:
                max_ratio = ratio
    
    outer_side = max_ratio
    
    # Create inner list with all hexagons at angle 0
    inner = [[x, y, 0.0] for (x, y) in positions]
    
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
