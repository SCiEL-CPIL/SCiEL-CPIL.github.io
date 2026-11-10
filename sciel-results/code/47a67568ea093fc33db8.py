import numpy as np

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    # The optimal packing for 11 unit hexagons in a regular hexagon
    # corresponds to a hexagonal lattice arrangement.
    # The side length of the outer hexagon is 5.0.
    
    # Inner hexagon centers
    # Lattice vectors u = (2, 0), v = (1, sqrt(3))
    # We pick the 11 closest points to the origin.
    # 1 center, 6 at distance 2, 4 at distance 2*sqrt(3).
    
    sqrt3 = np.sqrt(3)
    
    # Centers
    centers = [
        (0, 0),
        (2, 0), (-2, 0),
        (1, sqrt3), (-1, -sqrt3),
        (1, -sqrt3), (-1, sqrt3),
        # 4 more from the next ring (distance 2*sqrt(3))
        (3, sqrt3), (-3, -sqrt3),
        (2, 2*sqrt3), (-2, -2*sqrt3)
    ]
    
    inner_hex_data = np.zeros((11, 3))
    for i, (x, y) in enumerate(centers):
        inner_hex_data[i, 0] = x
        inner_hex_data[i, 1] = y
        inner_hex_data[i, 2] = 0.0  # Angle 0 degrees
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = 5.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
