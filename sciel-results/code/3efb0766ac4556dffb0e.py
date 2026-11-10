import numpy as np

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a 7-hex central cluster with 4 additional hexagons placed in the gaps,
    fitting exactly into a container of side length 4.
    """
    n = 11
    sqrt3 = np.sqrt(3)
    
    # 7 central hexagons
    centers = [
        [0, 0],
        [sqrt3, 0],
        [-sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [sqrt3/2, -1.5],
        [-sqrt3/2, -1.5]
    ]
    
    # 4 additional hexagons in the gaps at angles 30, 90, 210, 270 degrees
    # Distance from center is 3
    gap_centers = [
        [3 * np.cos(np.deg2rad(30)), 3 * np.sin(np.deg2rad(30))],
        [3 * np.cos(np.deg2rad(90)), 3 * np.sin(np.deg2rad(90))],
        [3 * np.cos(np.deg2rad(210)), 3 * np.sin(np.deg2rad(210))],
        [3 * np.cos(np.deg2rad(270)), 3 * np.sin(np.deg2rad(270))]
    ]
    
    all_centers = centers + gap_centers
    
    # All inner hexagons have angle 0
    inner_hex_data = np.array([(x, y, 0) for x, y in all_centers])
    
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 4.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
