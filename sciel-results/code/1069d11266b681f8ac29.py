import numpy as np

def construct():
    # Configuration: 1 center, 6 ring 1, 4 ring 2 "edge"
    # Axial coordinates
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (-1, -1)
    ]
    
    # Convert to Cartesian
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Generate vertices for all inner hexagons
    # Unit hexagon, pointy-top orientation
    # Vertices at angles 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    dx = np.cos(angles)
    dy = np.sin(angles)
    
    all_vertices = []
    for cx, cy in centers:
        for i in range(6):
            all_vertices.append((cx + dx[i], cy + dy[i]))
            
    all_vertices = np.array(all_vertices)
    
    # Find minimal enclosing hexagon side length S
    # Outer hexagon is regular, centered at (0,0)
    # We can rotate the outer hexagon.
    # The condition for a point (x,y) to be inside a hexagon of side S, 
    # rotated by angle alpha, is:
    # max over k=0..5 of (dot(p, n_k)) <= S * sqrt(3) / 2
    # where n_k are normals to the sides.
    # Normals for flat-top hexagon (vertices at 0, 60...) are at 30, 90...
    # Wait, if vertices are at 0, 60, the sides are perpendicular to 30, 90...
    # If we rotate the hexagon by alpha, the normals rotate by alpha.
    
    min_S = float('inf')
    # Try 360 angles? Or just optimize?
    # Due to symmetry of the cluster (not perfectly 6-fold), we should scan.
    # But 120s limit. 11 hexagons is small.
    
    # Let's try a few key angles
    for alpha_deg in range(0, 360, 5):
        alpha = np.deg2rad(alpha_deg)
        # Normals for a flat-top hexagon (vertices at 0, 60) are at 30, 90, 150, 210, 270, 330
        # Rotated by alpha:
        normal_angles = np.deg2rad([30, 90, 150, 210, 270, 330]) + alpha
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # Project all vertices onto these normals
        # p dot n
        projections = np.max(all_vertices @ np.column_stack((nx, ny)), axis=1)
        
        # max projection
        max_proj = np.max(projections)
        
        # S * sqrt(3)/2 >= max_proj
        S = max_proj * 2 / np.sqrt(3)
        
        if S < min_S:
            min_S = S
            best_alpha = alpha_deg
            
    # Return the best configuration
    # inner: list of [x, y, angle_degrees]
    # All inner hexagons are aligned with the grid (angle 0 relative to grid, but grid is pointy-top)
    # The task says "angle_degrees" for inner hexagons.
    # If we use pointy-top for calculation, the angle is 0?
    # Standard hexagon angle 0 usually means flat-top or pointy-top?
    # Let's assume 0 is flat-top (vertex at right).
    # My calculation used pointy-top (vertex at top).
    # So inner hexagons should be rotated by 90 degrees?
    # Or just define the angle consistently.
    # If I define inner hexagons as "pointy-top", their angle is 90 (if 0 is flat-top).
    # Let's use 0 for flat-top to be safe, and adjust coordinates?
    # No, the coordinates are fixed.
    # If the evaluator expects 0=flat-top, and I provide pointy-top coords,
    # I should report angle 90 for inner hexagons.
    
    inner_list = []
    for cx, cy in centers:
        inner_list.append([cx, cy, 90.0]) # 90 deg rotation from flat-top to pointy-top
        
    return inner_list, (0.0, 0.0), min_S, best_alpha

