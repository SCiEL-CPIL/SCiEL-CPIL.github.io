import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Define the centers of the 12 hexagons in a hexagonal lattice
    # 1 center, 6 in ring 1, 5 in ring 2 (consecutive to minimize bounding box)
    
    # Center
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 hexagons
    # Distance sqrt(3)
    r1 = np.sqrt(3)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 12 hexagons, we pick 5 consecutive ones
    # Angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    # We pick 5 consecutive, e.g., -60, -30, 0, 30, 60 (relative to x-axis)
    # To center the cluster, let's pick -60, -30, 0, 30, 60
    r2 = 2 * np.sqrt(3)
    angles_deg = [-60, -30, 0, 30, 60]
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        centers.append((x, y))
        
    inner_hex_data = np.zeros((12, 3))
    for i, (x, y) in enumerate(centers):
        inner_hex_data[i, 0] = x
        inner_hex_data[i, 1] = y
        inner_hex_data[i, 2] = 0.0 # Angle doesn't affect the bounding box size significantly if we calculate s correctly, but 0 is fine
        
    # Calculate outer hexagon side length
    # Max distance from center to any point in the cluster
    # Farthest points are vertices of the ring 2 hexagons
    # Distance to center of ring 2 hex is 2*sqrt(3)
    # Distance to vertex is 2*sqrt(3) + 1
    D = 2 * np.sqrt(3) + 1
    
    # The outer hexagon must contain these points.
    # Optimal rotation aligns vertices with the "corners" of the point distribution.
    # The points are at angles -60, -30, 0, 30, 60.
    # If we align outer hex vertices to -60, 0, 60, the points at -30, 30 are at apothems.
    # Required s: s * sqrt(3)/2 >= D  => s >= 2*D/sqrt(3)
    s = 2 * D / np.sqrt(3)
    
    # Outer hexagon centered at origin
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
