import numpy as np

def hexagon_packing_11():
    # Define 11 hexagons in a compact grid pattern
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2)
    ]
    
    # Convert to Cartesian (flat-top, side 1)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        centers.append((x, y))
    
    # Scale to ensure separation > 1e-6
    # Distance between centers in grid is sqrt(3).
    # We need distance > sqrt(3) + 1e-6.
    # Scale factor k such that k*sqrt(3) > sqrt(3) + 1e-6
    # k > 1 + 1e-6/sqrt(3)
    scale = 1.0 + 1e-5 / np.sqrt(3)
    
    scaled_centers = [(x*scale, y*scale) for x, y in centers]
    
    # Construct inner hex data
    # (x, y, angle_deg)
    inner_hex_data = np.zeros((11, 3))
    for i, (x, y) in enumerate(scaled_centers):
        inner_hex_data[i, 0] = x
        inner_hex_data[i, 1] = y
        inner_hex_data[i, 2] = 0.0 # angle
    
    # Determine required outer side length S
    # Outer hexagon is flat-top, centered at (0,0), angle 0.
    # Vertices of outer hexagon at distance S from origin.
    # Max x of inner vertices:
    # For a hexagon at (cx, cy) with side 1, max x is cx + 1.
    # Max y of inner vertices:
    # For a hexagon at (cx, cy), max y is cy + sqrt(3)/2 * 1? No.
    # Flat top hexagon vertices relative to center:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # So max x offset is 1.
    # Max y offset is sqrt(3)/2.
    
    max_x = max(cx + 1 for cx, cy in scaled_centers)
    max_y = max(cy + np.sqrt(3)/2 for cx, cy in scaled_centers)
    
    # Constraint 1: S >= max_x
    # Constraint 2: The top edge of the outer hexagon is at y = S * sqrt(3) / 2.
    # So S * sqrt(3) / 2 >= max_y  => S >= max_y * 2 / sqrt(3)
    
    S1 = max_x
    S2 = max_y * 2 / np.sqrt(3)
    
    S = max(S1, S2)
    
    # Add safety margin
    S += 0.01
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, S
