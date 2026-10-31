import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    axial = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (1, -2), (-1, 2)
    ]
    
    points = []
    for q, r in axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        points.append([x, y])
    points = np.array(points)
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Normals for flat-top hexagon (vertical sides? No, let's use standard 0,60...)
    # The set of normals is invariant under rotation, just shifted.
    # We can just rotate the points and check against fixed normals.
    for angle_deg in range(0, 31, 1):
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot_matrix = np.array([[c, -s], [s, c]])
        # Rotate points by -angle to align with container?
        # If container is rotated by angle, points are effectively rotated by -angle in container frame.
        rotated_points = points @ rot_matrix.T
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = np.deg2rad(60 * k)
            nx, ny = np.cos(normal_angle), np.sin(normal_angle)
            projections = rotated_points @ np.array([nx, ny])
            max_proj = max(max_proj, np.max(np.abs(projections)))
        
        side = 2.0 * max_proj / np.sqrt(3)
        
        if side < best_side:
            best_side = side
            best_angle = angle_deg
            
    inner = []
    for p in points:
        # Inner hexagons are not rotated relative to each other?
        # The task says "each inner hexagon may translate and rotate".
        # But for this compact packing, they are all aligned.
        # So angle is 0.
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    center = [0.0, 0.0]
    
    return inner, center, float(best_side), float(best_angle)
# EVOLVE-BLOCK-END
