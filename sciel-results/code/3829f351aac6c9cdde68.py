# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a subset of the 19-hexagon cluster (side 3 in tiling units) to minimize the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree).
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Define the 11 inner hexagon centers
    # 1 center
    # 6 inner ring
    # 4 outer ring (selected for symmetry)
    
    sqrt3 = np.sqrt(3)
    
    centers = [
        [0, 0],  # Center
        # Inner Ring (6)
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        # Outer Ring (4) - Selected from the 12 available in the 19-cluster
        # (2,0) -> (2*sqrt3, 0)
        [2*sqrt3, 0],
        # (0,2) -> (sqrt3, 3)
        [sqrt3, 3],
        # (-2,0) -> (-2*sqrt3, 0)
        [-2*sqrt3, 0],
        # (0,-2) -> (-sqrt3, -3)
        [-sqrt3, -3]
    ]
    
    inner_hex_data = np.array(centers + [[0]] * 11)  # Add angle 0
    
    outer_hex_data = np.array([0, 0, 0])
    
    # Calculate minimal side length
    # Max vertex distance from origin is 2*sqrt3 + 1
    outer_hex_side_length = 2 * sqrt3 + 1
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
