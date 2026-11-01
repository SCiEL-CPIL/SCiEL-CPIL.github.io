import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define centers for the 11 hexagons
    # Base 7 hexagons
    centers = [
        (0.0, 0.0),
        (np.sqrt(3), 0.0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (-np.sqrt(3), 0.0),
        (np.sqrt(3)/2, -1.5),
        (-np.sqrt(3)/2, -1.5)
    ]
    
    # Add 4 more to form a "cross"
    # (2*sqrt(3), 0), (0, 3), (-2*sqrt(3), 0), (0, -3)
    centers.append((2*np.sqrt(3), 0.0))
    centers.append((0.0, 3.0))
    centers.append((-2*np.sqrt(3), 0.0))
    centers.append((0.0, -3.0))
    
    # Convert to array
    C = np.array(centers)
    
    # The vertices of each unit hexagon (flat top)
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    offsets = np.column_stack((cos_a, sin_a))
    
    # All vertices of all hexagons
    # Shape: (N*6, 2)
    vertices = C[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    vertices = vertices.reshape(-1, 2)
    
    # Optimize rotation of enclosing hexagon
    # We want to minimize S such that max(v dot n_k) <= S * sqrt(3)/2
    # n_k are normals at phi + k*pi/3
    # Due to symmetry, we only need to check phi in [0, pi/6]
    
    best_phi = 0.0
    best_S = float('inf')
    
    # Sample phi
    for phi in np.linspace(0, np.pi/6, 100):
        normals = []
        for k in range(6):
            angle = phi + k * np.pi / 3
            normals.append((np.cos(angle), np.sin(angle)))
        normals = np.array(normals)
        
        # Project all vertices onto all normals
        # dots[i, j] = vertices[i] dot normals[j]
        dots = vertices @ normals.T
        max_proj = np.max(dots)
        
        S = max_proj * 2 / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_phi = phi
            
    # Refine with a few more steps if needed, but 100 steps is likely enough for this simple geometry
    # Let's do a finer search around best_phi
    phi_start = max(0, best_phi - np.pi/600)
    phi_end = min(np.pi/6, best_phi + np.pi/600)
    for phi in np.linspace(phi_start, phi_end, 100):
        normals = []
        for k in range(6):
            angle = phi + k * np.pi / 3
            normals.append((np.cos(angle), np.sin(angle)))
        normals = np.array(normals)
        dots = vertices @ normals.T
        max_proj = np.max(dots)
        S = max_proj * 2 / np.sqrt(3)
        if S < best_S:
            best_S = S
            best_phi = phi

    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    # We can keep inner hexes at angle 0
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    # outer: center (0,0), side length best_S, rotation best_phi (in degrees)
    outer_center = (0.0, 0.0)
    outer_side = best_S
    outer_angle = np.rad2deg(best_phi)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
