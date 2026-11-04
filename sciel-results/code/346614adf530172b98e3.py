import numpy as np

def construct():
    # Define the centers of the 11 hexagons
    # 1 center
    centers = [(0.0, 0.0)]
    
    # 6 neighbors
    for i in range(6):
        angle = np.pi / 3 * i
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
        
    # 4 corners
    # These are at distance 3 from center, at angles 30, 150, 210, 330
    corner_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for angle in corner_angles:
        x = 3 * np.cos(angle)
        y = 3 * np.sin(angle)
        centers.append((x, y))
        
    # Convert to array
    centers = np.array(centers)
    
    # The hexagons are unit side length.
    # We need to find the minimum side length L of a regular hexagon
    # that contains all these hexagons.
    # The outer hexagon can be rotated.
    
    # The condition for a hexagon with inradius R to contain a set of points
    # (vertices of the inner hexagons) is that the max projection of the
    # points onto any of the 6 normal directions is <= R.
    # The 6 normal directions are theta + k*pi/3.
    
    # We need to find the minimum R such that there exists a theta
    # where max_k max_p (p . n_k(theta)) <= R.
    
    # The vertices of each inner hexagon are at distance 1 from its center.
    # So for a center C, the max projection in direction n is C.n + 1.
    
    # We iterate over possible rotations theta of the outer hexagon.
    # Due to symmetry, we only need to check theta in [0, pi/6].
    
    min_R = float('inf')
    
    # Sample theta
    for theta in np.linspace(0, np.pi/6, 1000):
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0
        for n in normals:
            # Project all centers
            projs = centers @ n
            # Add 1 for the radius of the hexagons
            max_p = np.max(projs) + 1.0
            if max_p > max_proj:
                max_proj = max_p
        
        if max_proj < min_R:
            min_R = max_proj
    
    # L = R * 2 / sqrt(3)
    L = min_R * 2 / np.sqrt(3)
    
    # Construct the return values
    # inner: rows of (x, y, angle_degrees)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # center of outer hexagon
    outer_center = (0.0, 0.0)
    
    # side length
    outer_side = L
    
    # rotation of outer hexagon
    # We need to find the theta that gave min_R
    # Re-calculate to find best theta
    best_theta = 0
    min_R_val = float('inf')
    for theta in np.linspace(0, np.pi/6, 1000):
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0
        for n in normals:
            projs = centers @ n
            max_p = np.max(projs) + 1.0
            if max_p > max_proj:
                max_proj = max_p
        
        if max_proj < min_R_val:
            min_R_val = max_proj
            best_theta = theta
            
    outer_angle = np.degrees(best_theta)
    
    return inner, outer_center, outer_side, outer_angle
