# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice basis vectors for unit hexagons
    # Adjacent hexagon centers are at distance sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3 / 2.0, 1.5])
    
    # Generate lattice points and select the 11 closest to origin
    candidates = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            pos = i * a1 + j * a2
            dist = np.linalg.norm(pos)
            candidates.append((dist, pos[0], pos[1]))
    
    candidates.sort(key=lambda x: x[0])
    selected = candidates[:11]
    positions = np.array([[s[1], s[2]] for s in selected])
    
    # Compute required outer hexagon side length
    # For a regular hexagon centered at origin with angle 0 and circumradius R:
    # A point (x,y) is inside if dot((x,y), n_k) <= R * sqrt(3) / 2 for all 6 edge normals
    # The 6 inward normals point in directions 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each vertex of each inner hexagon, compute the max projection onto any normal
    max_dot = 0.0
    for pos in positions:
        cx, cy = pos
        # Unit hexagon vertices at (cx, cy) with angle 0
        vertex_angles = np.deg2rad(60 * np.arange(6))
        verts = np.column_stack([
            cx + np.cos(vertex_angles),
            cy + np.sin(vertex_angles)
        ])
        dots = verts @ normals.T
        max_dot = max(max_dot, float(np.max(dots)))
    
    # Required inradius = max_dot, circumradius R = max_dot * 2 / sqrt(3)
    outer_s = max_dot * 2.0 / sqrt3
    
    # Ensure outer_s >= sqrt(11)
    outer_s = max(outer_s, np.sqrt(11))
    
    inner_hex_data = np.column_stack([positions, np.zeros(11)])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_s


# EVOLVE-BLOCK-END
