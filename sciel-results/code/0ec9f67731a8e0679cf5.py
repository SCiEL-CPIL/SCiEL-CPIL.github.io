# EVOLVE-BLOCK-START
import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a 1-center + 6-ring + 5-outer-ring configuration.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    # 1. Center hexagon
    centers = []
    centers.append([0.0, 0.0, 0.0])
    
    # 2. First ring: 6 hexagons at distance sqrt(3)
    r1 = np.sqrt(3)
    for i in range(6):
        angle = i * 60
        rad = np.deg2rad(angle)
        x = r1 * np.cos(rad)
        y = r1 * np.sin(rad)
        centers.append([x, y, 0.0])
        
    # 3. Second ring: 5 hexagons
    # We use 5 of the 6 "corner" positions of the second ring.
    # The 6 positions are at distance 2*sqrt(3) from center, at angles 30, 90, 150, 210, 270, 330.
    # We skip one to have exactly 12 hexagons total.
    r2 = 2 * np.sqrt(3)
    # Angles: 30, 90, 150, 210, 270 (skipping 330)
    angles = [30, 90, 150, 210, 270]
    for angle in angles:
        rad = np.deg2rad(angle)
        x = r2 * np.cos(rad)
        y = r2 * np.sin(rad)
        centers.append([x, y, 0.0])
        
    inner_hex_data = np.array(centers)
    
    # Outer hexagon
    # The bounding box of the inner hexagons has width 8 (from x=-4 to x=4).
    # A regular hexagon with side s has width 2s.
    # So 2s >= 8 => s >= 4.
    # We set s = 4.
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = 4.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
