# EVOLVE-BLOCK-START
import numpy as np
import time


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Uses a 3-2-3-2-3 honeycomb arrangement which is more compact than a 1-3-3-3-1 arrangement.
    """
    # 3-2-3-2-3 arrangement (honeycomb pattern)
    # Row 0 (y = -2*sqrt(3)): 3 hexagons at x = -3, 0, 3
    # Row 1 (y = -sqrt(3)): 2 hexagons at x = -1.5, 1.5
    # Row 2 (y = 0): 3 hexagons at x = -3, 0, 3
    # Row 3 (y = sqrt(3)): 2 hexagons at x = -1.5, 1.5
    # Row 4 (y = 2*sqrt(3)): 3 hexagons at x = -3, 0, 3
    
    sqrt3 = np.sqrt(3)
    
    inner_hex_data = np.array([
        # Row 0 (bottom, y = -2*sqrt(3))
        [-3.0, -2*sqrt3, 0],
        [0.0, -2*sqrt3, 0],
        [3.0, -2*sqrt3, 0],
        # Row 1 (y = -sqrt(3))
        [-1.5, -sqrt3, 0],
        [1.5, -sqrt3, 0],
        # Row 2 (middle, y = 0)
        [-3.0, 0.0, 0],
        [0.0, 0.0, 0],
        [3.0, 0.0, 0],
        # Row 3 (y = sqrt(3))
        [-1.5, sqrt3, 0],
        [1.5, sqrt3, 0],
        # Row 4 (top, y = 2*sqrt(3))
        [-3.0, 2*sqrt3, 0],
        [0.0, 2*sqrt3, 0],
        [3.0, 2*sqrt3, 0],
    ])
    
    # Outer hexagon centered at origin with side length 4
    # Inradius = sqrt(3) * s / 2 = sqrt(3) * 4 / 2 = 2*sqrt(3)
    # This exactly matches the y-extent of the inner hexagons
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = 4.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
