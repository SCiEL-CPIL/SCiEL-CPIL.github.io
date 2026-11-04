import numpy as np

N = 12

def construct():
    # 3x4 grid of hexagons
    # Lattice vectors
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * u + j * v
            centers.append(c)
            
    centers = np.array(centers)
    
    # Center the grid
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Vertices of a unit hexagon at angle 0
    # Angles: 0, 60, 120, 180, 240, 300
    hex_angles = np.deg2rad(np.arange(0, 360, 60))
    hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    best_S = float('inf')
    best_angle = 0.0
    
    # Try rotating the outer hexagon
    # Due to symmetry, 0 to 60 is sufficient
    for deg in range(0, 61):
        alpha = np.deg2rad(deg)
        
        # Rotate vertices by -alpha to align with standard hexagon constraints
        # Standard hexagon constraints for side S:
        # |x| <= S
        # |y| <= S*sqrt(3)/2
        # |sqrt(3)x + y| <= S*sqrt(3)
        # |sqrt(3)x - y| <= S*sqrt(3)
        
        # We need to check all vertices of all inner hexagons
        # Inner hexagons are at angle 0 relative to their centers
        # So their vertices are centers + hex_vertices
        
        required_S = 0.0
        
        for c in centers:
            verts = c + hex_vertices
            # Rotate verts by -alpha
            cos_a = np.cos(-alpha)
            sin_a = np.sin(-alpha)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            verts_rot = verts @ R.T
            
            x = verts_rot[:, 0]
            y = verts_rot[:, 1]
            
            # Constraints
            s1 = np.max(np.abs(x))
            s2 = np.max(np.abs(y)) * 2 / np.sqrt(3)
            s3 = np.max(np.abs(np.sqrt(3)*x + y)) / np.sqrt(3)
            s4 = np.max(np.abs(np.sqrt(3)*x - y)) / np.sqrt(3)
            
            local_S = max(s1, s2, s3, s4)
            if local_S > required_S:
                required_S = local_S
                
        if required_S < best_S:
            best_S = required_S
            best_angle = deg
            
    # If best_S is still inf, something went wrong
    if best_S == float('inf'):
        best_S = 40.0
        best_angle = 0.0
        
    # Format inner list: (x, y, angle_degrees)
    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    
    return inner, (0.0, 0.0), float(best_S), float(best_angle)
