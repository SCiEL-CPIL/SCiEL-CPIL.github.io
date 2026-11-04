import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 1. Define the 11 centers in the hex lattice
    # Axial coordinates (q, r)
    centers_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers_cart = []
    for q, r in centers_axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers_cart.append((x, y))
        
    # 2. Generate vertices for each hexagon (angle 0)
    # Unit hexagon vertices relative to center (angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    hex_verts_rel = []
    for k in range(6):
        angle = np.deg2rad(60 * k)
        hex_verts_rel.append((np.cos(angle), np.sin(angle)))
        
    all_points = []
    for cx, cy in centers_cart:
        for vx, vy in hex_verts_rel:
            all_points.append((cx + vx, cy + vy))
            
    all_points = np.array(all_points)
    
    # 3. Compute Convex Hull
    hull = ConvexHull(all_points)
    hull_points = all_points[hull.vertices]
    
    # 4. Find optimal outer hexagon rotation and side length
    # The outer hexagon has 6 sides.
    # The distance from center to each side is A = S * sqrt(3) / 2.
    # The normal vectors to the sides are at angles theta + 30 + 60*k.
    # We need max(dot(p, n_k)) <= A for all k and all p in hull.
    # So A >= max_k (max_p (dot(p, n_k))).
    # S = 2 * A / sqrt(3).
    
    best_S = float('inf')
    best_theta = 0.0
    
    # Scan theta from 0 to 60 degrees
    # Due to symmetry of the hull? The hull is not necessarily 6-fold symmetric,
    # but the outer hexagon is.
    # We just need to find the min over theta.
    thetas = np.linspace(0, np.pi/3, 1000)
    
    for theta in thetas:
        # Normals for the 6 sides
        # Side 0 normal at theta + 30 deg
        # Side k normal at theta + 30 + 60*k deg
        max_projection = 0.0
        for k in range(6):
            angle_n = theta + np.pi/6 + k * np.pi/3
            n_x = np.cos(angle_n)
            n_y = np.sin(angle_n)
            # Project all hull points onto this normal
            projections = hull_points[:, 0] * n_x + hull_points[:, 1] * n_y
            proj_max = np.max(projections)
            if proj_max > max_projection:
                max_projection = proj_max
        
        A = max_projection
        S = 2 * A / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_theta = theta
            
    # The return interface requires:
    # inner: rows (x, y, angle_degrees)
    # center: (x, y)
    # side: S
    # angle: rotation of outer hexagon
    
    inner = []
    for cx, cy in centers_cart:
        inner.append([cx, cy, 0.0])
        
    return inner, (0.0, 0.0), best_S, np.rad2deg(best_theta)
# EVOLVE-BLOCK-END
