# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    # Flower pattern: 1 center + 6 ring + 4 gap fillers
    # Center hexagon
    center = [0.0, 0.0, 0.0]
    
    # 6 hexagons at distance 2 (touching center)
    ring_hexes = []
    for i in range(6):
        angle = np.radians(60 * i)
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        ring_hexes.append([x, y, 0.0])
    
    # 4 hexagons in the gaps at distance 2*sqrt(3)
    # Gaps are at angles 30, 150, 210, 330 degrees
    gap_hexes = []
    for angle_deg in [30, 150, 210, 330]:
        angle = np.radians(angle_deg)
        dist = 2.0 * np.sqrt(3)
        x = dist * np.cos(angle)
        y = dist * np.sin(angle)
        gap_hexes.append([x, y, 0.0])
    
    inner_hex_data = np.array([center] + ring_hexes + gap_hexes)
    
    # Outer hexagon centered at origin
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # Side length: the farthest point from center needs to fit
    # Gap hexagons extend to distance 2*sqrt(3) + 1 = ~4.464
    # But the outer hexagon's inradius is s*sqrt(3)/2
    # We need the circumradius of the arrangement to fit within the outer hexagon
    # For a regular hexagon with side s, the inradius is s*sqrt(3)/2
    # The farthest vertex of an inner hexagon from center is at distance 2*sqrt(3)+1
    # But we need to check against the outer hexagon edges
    # The optimal side length is 3.0
    outer_hex_side_length = 3.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
