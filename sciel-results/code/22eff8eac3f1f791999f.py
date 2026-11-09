import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 1. Generate Centers for 3x4 Grid of Pointy-Top Hexagons
    # Pointy-Top: Width sqrt(3), Height 2
    # Spacing: dx = sqrt(3), dy = 1.5, shift = sqrt(3)/2
    
    sqrt3 = np.sqrt(3)
    centers = []
    # 3 rows, 4 cols
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 + r * (sqrt3 / 2.0)
            y = r * 1.5
            centers.append((x, y))
            
    # 2. Generate Vertices
    # Unit Pointy-Top Hex Vertices relative to center
    # Angle 90 deg from standard (vertex at 0)
    angles = np.deg2rad(np.array([90, 30, -30, -90, -150, 150]))
    base_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    
    all_vertices = []
    for cx, cy in centers:
        vs = base_vertices + np.array([cx, cy])
        all_vertices.append(vs)
        
    all_vertices = np.vstack(all_vertices) # Shape (72, 2)
    
    # 3. Determine Center
    center = np.mean(all_vertices, axis=0)
    
    # 4. Optimize Container Rotation and Size
    # Container Hexagon defined by center, side L, rotation theta.
    base_normal_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    base_normals = np.column_stack((np.cos(base_normal_angles), np.sin(base_normal_angles)))
    
    def calc_min_side(theta_deg):
        theta = np.deg2rad(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        normals = (rot_mat @ base_normals.T).T
        
        verts_rel = all_vertices - center
        proj = verts_rel @ normals.T
        max_proj = np.max(proj)
        
        return 2.0 * max_proj / sqrt3

    res = minimize_scalar(calc_min_side, bounds=(0, 90), method='bounded')
    
    best_theta = res.x
    best_side = res.fun
    
    # 5. Construct Return Values
    inner = [[cx, cy, 90.0] for cx, cy in centers]
    
    return inner, (center[0], center[1]), best_side, best_theta
# EVOLVE-BLOCK-END
