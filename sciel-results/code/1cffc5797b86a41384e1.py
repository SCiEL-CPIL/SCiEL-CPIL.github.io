import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for 12 hexagons in a hexagonal shell (Layer 2)
    # Lattice spacing is sqrt(3)
    # Axial coordinates (q, r) to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    centers = []
    # Layer 2 points in axial coordinates
    # Max distance 2 from origin in hex grid
    # Condition: max(abs(q), abs(r), abs(q+r)) == 2
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                x = np.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                centers.append((x, y))
    
    # There should be 12 centers
    assert len(centers) == 12, f"Expected 12 centers, got {len(centers)}"
    
    # Function to get vertices of a unit hexagon centered at (cx, cy) with rotation alpha
    def get_hex_vertices(cx, cy, alpha):
        # Unit hexagon vertices (side 1)
        # Angles: 0, 60, 120, 180, 240, 300 degrees
        # Radius (center to vertex) is 1
        angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + alpha)
        vx = np.cos(angles)
        vy = np.sin(angles)
        return np.column_stack((vx + cx, vy + cy))
    
    # Objective function: Minimize outer side length S
    # Outer hexagon is centered at (0,0) with rotation 0 (flat top? or pointy top?)
    # Let's assume standard orientation: vertices at 0, 60, ...
    # Normals to sides are at 30, 90, 150, ...
    # Side length S corresponds to apothem a = S * sqrt(3) / 2
    # Containment: For all points p, |p . n_i| <= a for i in 0,1,2
    # So S = max_i |p . n_i| / (sqrt(3)/2) = 2/sqrt(3) * max_i |p . n_i|
    
    # Normals for outer hexagon (rotation 0)
    # If vertices are at 0 deg, side 1 is between 0 and 60. Midpoint angle 30.
    # Normal is at 30 deg.
    # Next side midpoint 90 deg. Normal 90 deg.
    # Next side midpoint 150 deg. Normal 150 deg.
    normal_angles = np.deg2rad([30, 90, 150])
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    def objective(alpha):
        alpha = np.deg2rad(alpha)
        all_points = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy, alpha)
            all_points.append(verts)
        
        all_points = np.vstack(all_points)
        
        # Project onto normals
        projections = all_points @ normals.T
        max_proj = np.max(np.abs(projections))
        
        S = (2.0 / np.sqrt(3.0)) * max_proj
        return S
    
    # Optimize alpha
    # Due to symmetry, we only need to search [0, 60]
    res = minimize_scalar(objective, bounds=(0, 60), method='bounded')
    optimal_alpha = res.x
    
    # Construct inner list
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, optimal_alpha])
    
    # Outer hexagon
    outer_center = (0.0, 0.0)
    side_length = res.fun
    outer_angle = 0.0
    
    return inner, outer_center, side_length, outer_angle
# EVOLVE-BLOCK-END
