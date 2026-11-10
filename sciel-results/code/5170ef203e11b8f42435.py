import numpy as np

def hexagon_packing_12():
    # Correct lattice vectors for honeycomb packing of unit hexagons
    # Neighbor vectors are at angles 30, 90, 150, 210, 270, 330 degrees
    # Distance is sqrt(3)
    # e1: angle 30 deg -> (sqrt(3)*cos(30), sqrt(3)*sin(30)) = (1.5, sqrt(3)/2)
    # e2: angle 90 deg? No, let's use two basis vectors that span the lattice.
    # The neighbors are e1, e2, e1-e2, -e1, -e2, e2-e1?
    # Let's pick:
    # u1 = (1.5, sqrt(3)/2)  (angle 30)
    # u2 = (-1.5, sqrt(3)/2) (angle 150)
    # u1 + u2 = (0, sqrt(3)) (angle 90)
    # u1 - u2 = (3, 0) (angle 0) -> This is distance 3? No.
    # Distance between centers at 0 and u1-u2 is 3?
    # u1-u2 = (3, 0). Distance 3.
    # But the neighbor at angle 0 is at distance sqrt(3)?
    # Wait. The 6 neighbors are at distance sqrt(3).
    # Angles: 30, 90, 150, 210, 270, 330.
    # Vector at 30: (1.5, sqrt(3)/2).
    # Vector at 90: (0, sqrt(3)).
    # Vector at 150: (-1.5, sqrt(3)/2).
    # Vector at 210: (-1.5, -sqrt(3)/2).
    # Vector at 270: (0, -sqrt(3)).
    # Vector at 330: (1.5, -sqrt(3)/2).
    
    # Basis vectors for grid generation:
    # We can use u1 = (1.5, sqrt(3)/2) and u2 = (-1.5, sqrt(3)/2).
    # Grid points: i*u1 + j*u2.
    # This generates the triangular lattice.
    
    u1 = np.array([1.5, np.sqrt(3) / 2.0])
    u2 = np.array([-1.5, np.sqrt(3) / 2.0])
    
    # Generate 12 centers in a 3x4 parallelogram grid
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * u1 + j * u2
            centers.append(c)
    centers = np.array(centers)
    
    # Center the cluster
    centers -= centers.mean(axis=0)
    
    # Scale to ensure separation > 1e-6
    # The minimum distance between non-overlapping hexagons is 0 (touching).
    # We need a gap > 1e-6.
    # Scaling by (1 + eps) creates a gap of approx eps * sqrt(3) between touching edges.
    # We need eps * sqrt(3) > 1e-6 => eps > 5.77e-7.
    # Use 1e-6 to be safe.
    scale = 1.0 + 1e-6
    centers *= scale
    
    # Compute the minimal side length s for an outer hexagon aligned with the lattice (angle 0)
    # The outer hexagon has center (0,0) and angle 0.
    # Its edge normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    
    normal_angles = np.pi/6 + np.arange(6) * np.pi/3
    normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
    
    # Vertices of a unit hexagon centered at origin
    vertex_angles = np.arange(6) * np.pi/3
    unit_vertices = np.array([[np.cos(a), np.sin(a)] for a in vertex_angles])
    
    max_proj = 0.0
    for c in centers:
        # For each center, check all its vertices
        for u in unit_vertices:
            v = c + u
            for n in normals:
                proj = np.dot(v, n)
                if proj > max_proj:
                    max_proj = proj
    
    # The required apothem is max_proj
    # Apothem = s * sqrt(3) / 2
    s = 2.0 * max_proj / np.sqrt(3)
    
    # Construct output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s
