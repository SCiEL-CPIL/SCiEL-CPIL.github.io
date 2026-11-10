import numpy as np
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a 'flower' configuration: 1 center, 6 in first ring, 5 in second ring (diagonal positions).
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    # Time guard
    start_time = time.time()
    deadline = 600.0
    
    # Inner hexagons
    # 1. Center
    centers = [[0.0, 0.0]]
    
    # 2. Ring 1 (6 hexes)
    # Distance sqrt(3), angles 0, 60, 120, 180, 240, 300
    r1 = np.sqrt(3)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        centers.append([x, y])
        
    # 3. Ring 2 (5 hexes)
    # Distance 3, angles 30, 90, 150, 210, 270
    r2 = 3.0
    for i in range(5):
        angle = np.deg2rad(30 + 60 * i)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        centers.append([x, y])
        
    inner_hex_data = np.array(centers)
    # All angles are 0
    inner_hex_data = np.column_stack((inner_hex_data, np.zeros(12)))
    
    # Outer hexagon
    # Side length 4
    # Rotated by 30 degrees to align vertices with inner vertices
    outer_hex_data = np.array([0.0, 0.0, 30.0])
    outer_hex_side_length = 4.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
