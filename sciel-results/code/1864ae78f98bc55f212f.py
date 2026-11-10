# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice arrangement: 4-3-4-1 rows
    a = sqrt3  # horizontal spacing
    b = 1.5    # vertical spacing
    
    positions = []
    # Row 0: 4 hexagons at y=0
    for i in range(4):
        positions.append((-1.5*a + i*a, 0.0))
    # Row 1: 3 hexagons at y=b, offset by a/2
    for i in range(3):
        positions.append((-a + i*a, b))
    # Row 2: 4 hexagons at y=2*b
    for i in range(4):
        positions.append((-1.5*a + i*a, 2.0*b))
    # Row 3: 1 hexagon at y=3*b
    positions.append((0.0, 3.0*b))
    
    positions = np.array(positions)
    
    # Center the arrangement at origin
    cx = np.mean(positions[:, 0])
    cy = np.mean(positions[:, 1])
    positions[:, 0] -= cx
    positions[:, 1] -= cy
    
    inner_hex_data = np.column_stack([positions, np.zeros(12)])
    
    # Get all vertices of all inner hexagons (72 vertices total)
    all_vertices = []
    for i in range(12):
        px, py, _ = inner_hex_data[i]
        for j in range(6):
            theta = np.radians(60 * j)
            vx = px + np.cos(theta)
            vy = py + np.sin(theta)
            all_vertices.append([vx, vy])
    all_vertices = np.array(all_vertices)
    
    # Find minimal outer hexagon side length by optimizing angle
    def compute_min_s(alpha_deg):
        alpha = np.radians(alpha_deg)
        max_s = 0.0
        for px, py in all_vertices:
            for i in range(6):
                theta = alpha + np.radians(60 * i + 30)
                nx = np.cos(theta)
                ny = np.sin(theta)
                required_s = (px * nx + py * ny) * 2.0 / sqrt3
                if required_s > max_s:
                    max_s = required_s
        return max(0.0, max_s)
    
    # Search over outer hexagon angle
    best_s = float('inf')
    best_angle = 0.0
    for angle_deg in np.linspace(0, 60, 121):
        s = compute_min_s(angle_deg)
        if s < best_s:
            best_s = s
            best_angle = angle_deg
    
    outer_hex_side_length = best_s
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
