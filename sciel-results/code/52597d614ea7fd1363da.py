import numpy as np

N = 12

def construct():
    # Define centers for a 3x4 grid of hexagons
    # Hexagon side length 1
    # Distance between centers in same row: sqrt(3)
    # Distance between centers in adjacent rows: 3/2 (vertical), sqrt(3)/2 (horizontal offset)
    
    centers = []
    # 3 rows, 4 columns
    # Row 0: y = -1.5
    # Row 1: y = 0
    # Row 2: y = 1.5
    
    # To make it symmetric, we can adjust the x-offsets.
    # Let's generate a 3x4 block.
    for r in range(3):
        for c in range(4):
            y = (r - 1) * 1.5
            # Offset every other row by sqrt(3)/2
            x_offset = (c * np.sqrt(3))
            if r % 2 == 1:
                x_offset += np.sqrt(3) / 2
            # Center the block
            # The x range for r=0 is [0, 3*sqrt(3)]
            # The x range for r=1 is [sqrt(3)/2, 3*sqrt(3) + sqrt(3)/2]
            # Midpoint of r=0: 1.5*sqrt(3)
            # Midpoint of r=1: 1.5*sqrt(3) + sqrt(3)/2
            # This is not centered.
            # Let's just generate them and then center the whole set.
            centers.append([x_offset, y])
            
    centers = np.array(centers)
    # Center the configuration
    center = np.mean(centers, axis=0)
    centers -= center
    
    # Generate vertices for all 12 hexagons
    # Hexagon vertices relative to center (side 1, flat-top? or pointy-top?)
    # Standard math: vertices at angles 0, 60, ...
    # Let's use vertices at angles 0, 60, 120, 180, 240, 300
    # This gives a "pointy-top" if y is up?
    # Vertex 0: (1, 0). Vertex 1: (0.5, sqrt(3)/2).
    # This is a hexagon with a vertex on the x-axis.
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    verts_rel = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_verts = []
    for c in centers:
        all_verts.append(c + verts_rel)
    all_verts = np.vstack(all_verts)
    
    # Optimize outer hexagon rotation
    # The outer hexagon has side S.
    # Condition: for all vertices v, and for the 3 normal vectors n_k of the outer hexagon,
    # | n_k . (v - center) | <= S * sqrt(3) / 2
    # So S >= 2/sqrt(3) * max( | n_k . (v - center) | )
    
    def min_side(theta):
        # Outer hexagon normals
        # If outer hexagon is rotated by theta, its normals are at theta + 30, theta + 150, theta + 270?
        # A regular hexagon has normals at angles alpha, alpha+60, ..., alpha+300.
        # The constraint is defined by the 3 unique directions (opposites are same line).
        # Let's define the 3 normals as:
        # n0 = (cos(theta), sin(theta))
        # n1 = (cos(theta+60), sin(theta+60))
        # n2 = (cos(theta+120), sin(theta+120))
        # Wait, the apothem is perpendicular to the side.
        # If the side is horizontal, normal is vertical.
        # Let's just use 3 vectors separated by 60 degrees.
        
        normals = []
        for k in range(3):
            ang = theta + k * np.pi / 3
            normals.append([np.cos(ang), np.sin(ang)])
        normals = np.array(normals)
        
        # Project all vertices onto these normals
        # v is (72, 2), normals is (3, 2)
        # We want max | v . n | for all v, n
        # Use broadcasting
        # v @ normals.T -> (72, 3)
        projections = np.abs(all_verts @ normals.T)
        max_proj = np.max(projections)
        S = max_proj * 2 / np.sqrt(3)
        return S

    # Minimize S over theta in [0, pi/3)
    # Use simple grid search or scipy minimize
    from scipy.optimize import minimize_scalar
    
    res = minimize_scalar(min_side, bounds=(0, np.pi/3), method='bounded')
    theta_opt = res.x
    S_opt = res.fun
    
    # The inner hexagons are at 'centers' with angle 0
    # The outer hexagon is at 'center' (which is 0,0) with side S_opt and angle theta_opt
    # Note: The angle returned should be the rotation of the outer hexagon.
    # The interface expects: inner rows (x, y, angle_degrees), outer center, side, angle
    
    inner = [[c[0], c[1], 0.0] for c in centers]
    outer_center = [0.0, 0.0]
    outer_side = S_opt
    outer_angle = np.rad2deg(theta_opt)
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from scipy.optimize import minimize_scalar

    N = 12
    
    # Define centers for a 3x4 grid of hexagons
    centers = []
    for r in range(3):
        for c in range(4):
            y = (r - 1) * 1.5
            x_offset = (c * np.sqrt(3))
            if r % 2 == 1:
                x_offset += np.sqrt(3) / 2
            centers.append([x_offset, y])
            
    centers = np.array(centers)
    center = np.mean(centers, axis=0)
    centers -= center
    
    # Generate vertices for all 12 hexagons
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    verts_rel = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_verts = []
    for c in centers:
        all_verts.append(c + verts_rel)
    all_verts = np.vstack(all_verts)
    
    def min_side(theta):
        normals = []
        for k in range(3):
            ang = theta + k * np.pi / 3
            normals.append([np.cos(ang), np.sin(ang)])
        normals = np.array(normals)
        
        projections = np.abs(all_verts @ normals.T)
        max_proj = np.max(projections)
        S = max_proj * 2 / np.sqrt(3)
        return S

    res = minimize_scalar(min_side, bounds=(0, np.pi/3), method='bounded')
    theta_opt = res.x
    S_opt = res.fun
    
    inner = [[c[0], c[1], 0.0] for c in centers]
    outer_center = [0.0, 0.0]
    outer_side = S_opt
    outer_angle = np.rad2deg(theta_opt)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
