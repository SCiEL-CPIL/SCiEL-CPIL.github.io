import numpy as np

N = 12

def construct():
    # Axial coordinates for the 12 hexagons
    # 1 Center
    # 6 Inner Ring
    # 5 Outer Ring (Edge positions, skipping one to make 12 total)
    
    axial_coords = [
        (0, 0), # Center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), # Inner Ring
        (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2) # 5 Edge positions
    ]
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Calculate minimum side length S
    # Normals for the 6 sides of the container hexagon
    # Container is regular hexagon. 
    # If we align the container with the grid (flat-to-flat), the normals are at 0, 60, ...
    # Wait, the inner hexagons are axis-aligned (angle 0).
    # So their flat sides are at 0, 60, ...
    # The container should also be aligned to minimize S.
    # Normals angles: 0, 60, 120, 180, 240, 300 degrees.
    
    max_proj = 0.0
    for angle_deg in range(0, 360, 60):
        angle_rad = np.deg2rad(angle_deg)
        nx = np.cos(angle_rad)
        ny = np.sin(angle_rad)
        
        for cx, cy in centers:
            proj = cx * nx + cy * ny
            # The inner hexagon extends by apothem (sqrt(3)/2) in this direction
            # The container side S has apothem (sqrt(3)/2 * S)
            # So: proj + sqrt(3)/2 <= sqrt(3)/2 * S
            # S >= (proj + sqrt(3)/2) / (sqrt(3)/2)
            # S >= proj * (2/sqrt(3)) + 1
            required_s = proj * (2.0 / np.sqrt(3.0)) + 1.0
            if required_s > max_proj:
                max_proj = required_s
                
    S = max_proj
    
    # Prepare output
    # inner: list of [x, y, angle_degrees]
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    center = [0.0, 0.0]
    side = S
    angle = 0.0
    
    return inner, center, side, angle
