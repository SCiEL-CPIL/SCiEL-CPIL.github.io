# EVOLVE-BLOCK-START
import numpy as np
import time


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = 600.0  # Safety margin before 630s kill

    n = 12
    sqrt3 = np.sqrt(3)

    # Honeycomb arrangement with small gap
    # Scale factor to ensure > 1e-6 separation
    gap_factor = 1.0 + 1e-5  # Small expansion to ensure separation

    # Axial coordinates (q, r) for 12 hexagons in a 4x3 grid
    centers = []
    for q in range(4):
        for r in range(3):
            x = 1.5 * gap_factor * q
            y = sqrt3 * gap_factor * (r + q / 2.0)
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the configuration
    cx_mean = np.mean(centers[:, 0])
    cy_mean = np.mean(centers[:, 1])
    centers[:, 0] -= cx_mean
    centers[:, 1] -= cy_mean
    
    inner_hex_data = np.column_stack([centers, np.zeros(12)])

    # Find minimal regular hexagon (flat-top, angle=0) containing all vertices
    max_proj = 0.0
    for i in range(12):
        cx_i, cy_i = centers[i, 0], centers[i, 1]
        for j in range(6):
            angle_j = np.radians(60 * j)
            px = cx_i + np.cos(angle_j)
            py = cy_i + np.sin(angle_j)
            for k in range(6):
                angle_k = np.radians(60 * k + 30)
                proj = px * np.cos(angle_k) + py * np.sin(angle_k)
                if proj > max_proj:
                    max_proj = proj
    
    # s * sqrt(3) / 2 >= max_proj => s >= 2 * max_proj / sqrt(3)
    outer_hex_side_length = 2.0 * max_proj / sqrt3
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
