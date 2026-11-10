# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons inside a regular hexagon of side 5.
    Uses a hexagonal cluster arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    sqrt3 = np.sqrt(3.0)
    
    # Axial coordinates for 12 hexagons
    center = (0, 0)
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Compute all ring-2 axial coordinates
    ring2_all = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) == 2:
                ring2_all.append((q, r))
    
    # ring2_all has 12 elements. Pick every other to avoid adjacency.
    selected_ring2 = [ring2_all[i] for i in range(0, len(ring2_all), 2)]
    
    all_axial = [center] + ring1 + selected_ring2
    
    # Convert to Cartesian
    centers = []
    for q, r in all_axial:
        x = 3.0 * q + 1.5 * r
        y = (3.0 * sqrt3 / 2.0) * r
        centers.append((x, y))
    
    inner_hex_data = np.array([[x, y, 0.0] for x, y in centers])
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = 5.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
