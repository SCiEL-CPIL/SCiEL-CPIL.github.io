# EVOLVE-BLOCK-START
import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a triangular lattice arrangement with 1 center, 6 in the first ring, and 5 in the second ring.
    Optimizes the angle of the outer hexagon to minimize its side length.
    """
    # Define the 12 inner hexagon centers
    # Lattice spacing is sqrt(3)
    sqrt3 = np.sqrt(3)
    
    centers = []
    # Center
    centers.append([0.0, 0.0])
    
    # First ring (6 hexagons)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append([x, y])
        
    # Second ring (5 hexagons, skipping one to make 12 total)
    # We skip the one at 300 degrees (or any, due to symmetry of the outer optimization)
    for i in range(5):
        angle = np.deg2rad(60 * i)
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        centers.append([x, y])
        
    centers = np.array(centers)
    
    # Generate vertices for all inner hexagons
    # Each hexagon has side 1, angle 0
    # Vertices are at angles 0, 60, 120, 180, 240, 300 relative to center
    hex_angles = np.deg2rad(np.arange(0, 360, 60))
    cos_h = np.cos(hex_angles)
    sin_h = np.sin(hex_angles)
    
    # vertices shape: (12, 6, 2)
    vertices = np.zeros((len(centers), 6, 2))
    for i, c in enumerate(centers):
        vertices[i, :, 0] = c[0] + cos_h
        vertices[i, :, 1] = c[1] + sin_h
        
    # Flatten vertices to (72, 2)
    verts_flat = vertices.reshape(-1, 2)
    
    # Optimize outer hexagon angle
    min_s = np.inf
    best_angle = 0.0
    
    # The outer hexagon is symmetric every 60 degrees, so we search 0 to 60
    # Fine step for optimization
    for theta_deg in np.arange(0, 60.01, 0.1):
        theta_rad = np.deg2rad(theta_deg)
        # Edge normals are at theta + 30 + 60*k
        normal_angles = theta_rad + np.deg2rad(30) + np.deg2rad(60) * np.arange(6)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Compute dot products for all vertices with all normals
        # verts_flat: (72, 2), normals: (6, 2)
        # We need max over k of dot(v, n_k)
        # Result shape (72, 6)
        dots = verts_flat @ normals.T
        
        # Max dot product for each vertex across all normals
        max_dots = np.max(dots, axis=1)
        
        # Global max
        global_max_dot = np.max(max_dots)
        
        # Required s: global_max_dot <= s * sqrt(3) / 2
        s_candidate = global_max_dot * 2 / sqrt3
        
        if s_candidate < min_s:
            min_s = s_candidate
            best_angle = theta_deg
            
    # Construct the output arrays
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0 # All inner hexagons aligned with angle 0
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    return inner_hex_data, outer_hex_data, min_s

# EVOLVE-BLOCK-END
