import numpy as np
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # 1. Define the rigid cluster topology
    # We use a 3x4 parallelogram of hexagons
    # Basis vectors for honeycomb (flat-top hexagons)
    # Distance between centers is sqrt(3)
    u = np.array([math.sqrt(3), 0.0])
    v = np.array([math.sqrt(3)/2.0, 1.5])
    
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * u + j * v
            centers.append(c)
            
    centers = np.array(centers)
    
    # 2. Generate vertices for one unit hexagon
    # Side length 1
    # Vertices at angles 0, 60, ...
    hex_verts_local = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)
    ])
    
    # 3. Generate all vertices for the cluster
    # For each center, add the local vertices
    all_vertices = []
    for c in centers:
        verts = hex_verts_local + c
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices) # Shape (72, 2)
    
    # Center the cluster
    centroid = np.mean(all_vertices, axis=0)
    all_vertices_centered = all_vertices - centroid

    # 4. Optimize rotation
    best_side = float('inf')
    best_angle = 0.0
    
    # Search rotation from 0 to 60 degrees (hexagon symmetry)
    # Step size 0.1 degrees for good resolution
    angles = np.arange(0, 60, 0.1)
    
    # Precompute normals for outer hexagon (orientation 0)
    # Normals at 0, 60, 120, 180, 240, 300
    normals = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)
    ])
    
    for phi_deg in angles:
        if time.time() > deadline:
            break
            
        phi_rad = math.radians(phi_deg)
        c_phi = math.cos(phi_rad)
        s_phi = math.sin(phi_rad)
        
        # Rotation matrix
        R = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
        
        # Rotate vertices
        rotated_verts = all_vertices_centered @ R.T
        
        # Compute projections onto 6 normals
        # Shape (72, 6)
        projections = rotated_verts @ normals.T
        
        # Compute min and max for each direction
        mins = np.min(projections, axis=0)
        maxs = np.max(projections, axis=0)
        
        widths = maxs - mins
        
        # Required side length S = max(width) / sqrt(3)
        # Because apothem a = width/2, and S = 2a/sqrt(3) = width/sqrt(3)
        current_side = np.max(widths) / math.sqrt(3)
        
        if current_side < best_side:
            best_side = current_side
            best_angle = phi_deg

    # 5. Construct the result
    # We need to return the positions and angles of the 12 inner hexagons
    # And the outer hexagon center, side, angle
    
    # The inner hexagons are at the rotated centers
    phi_rad = math.radians(best_angle)
    c_phi = math.cos(phi_rad)
    s_phi = math.sin(phi_rad)
    R = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
    
    rotated_centers = centers @ R.T
    
    # The outer hexagon is centered at (0,0) in our calculation
    # We can return (0,0) as the center
    # The side length is best_side
    # The angle is 0 (since we optimized for outer angle 0)
    
    inner = []
    for c in rotated_centers:
        # Angle in degrees
        inner.append([c[0], c[1], best_angle])
        
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
