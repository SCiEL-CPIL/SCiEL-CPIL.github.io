import numpy as np
from itertools import product

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Unit hexagon vertices (flat top, centered at origin)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    hex_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Lattice vectors for flat-top hexagons
    # Distance between centers is sqrt(3)
    v1 = np.array([np.sqrt(3), 0.0])
    v2 = np.array([np.sqrt(3)/2.0, 1.5])
    
    # Generate candidate configurations
    # We will generate centers on a grid and select 12
    # To keep it simple, we define specific shapes
    
    candidates = []
    
    # Shape 1: 3x4 Parallelogram
    # i in 0..2, j in 0..3
    centers_1 = []
    for i in range(3):
        for j in range(4):
            centers_1.append(i * v1 + j * v2)
    candidates.append(np.array(centers_1))
    
    # Shape 2: 2x6 Parallelogram
    centers_2 = []
    for i in range(2):
        for j in range(6):
            centers_2.append(i * v1 + j * v2)
    candidates.append(np.array(centers_2))
    
    # Shape 3: 4x3 Parallelogram (swapped)
    centers_3 = []
    for i in range(4):
        for j in range(3):
            centers_3.append(i * v1 + j * v2)
    candidates.append(np.array(centers_3))
    
    # Shape 4: Hexagonal cluster (1 + 6 + 5)
    # Center
    c = [np.array([0.0, 0.0])]
    # Ring 1
    for k in range(6):
        ang = np.deg2rad(60 * k)
        c.append(np.array([np.cos(ang), np.sin(ang)]) * np.sqrt(3))
    # Ring 2 (partial)
    # We need 5 more.
    # Let's pick 5 from the 12 neighbors of Ring 1.
    # To be symmetric, maybe pick 5 that are "close" to each other?
    # Or just try a few combinations.
    # Let's try picking 5 consecutive ones to form a "fan".
    # Or better, try to minimize the bounding box.
    # Let's just generate a few specific subsets of Ring 2.
    ring2_centers = []
    for k in range(12):
        # Ring 2 centers are at distance 2*sqrt(3) from origin?
        # No, distance from origin to Ring 2 center is 2 * sqrt(3) ?
        # Let's calculate properly.
        # Ring 1 center at angle 0: (sqrt(3), 0).
        # Its neighbors in Ring 2 are at:
        # (sqrt(3)+sqrt(3), 0) = (2sqrt(3), 0)
        # (sqrt(3)+sqrt(3)/2, 1.5) = (1.5sqrt(3), 1.5)
        # etc.
        # Let's just generate all lattice points within distance 2.5 * sqrt(3) from origin
        # and select those at distance approx 2 * sqrt(3).
        pass # This is getting complex. Let's stick to parallelograms and maybe a 2x2x3 block?
        
    # Shape 5: 2x2x3? No, 2D.
    # Let's try a "Staircase"
    # Or just evaluate the parallelograms first.
    
    best_S = float('inf')
    best_config = None
    best_rotation = 0.0
    
    # Normals for outer hexagon (flat top, aligned)
    # Sides are vertical? No, flat top means horizontal top/bottom.
    # Normals to sides:
    # Top/Bottom sides: Normal is vertical (0, 1) and (0, -1).
    # Side 1 (top right): Normal at 30 deg?
    # Vertices at 0, 60, 120...
    # Side midpoints at 30, 90, 150...
    # Normals at 30, 90, 150, 210, 270, 330.
    # We only need 3 independent normals for the hexagon constraints (opposite sides are parallel).
    # n0: 90 deg (0, 1)
    # n1: 210 deg (-sqrt(3)/2, -0.5)
    # n2: 330 deg (sqrt(3)/2, -0.5)
    n0 = np.array([0.0, 1.0])
    n1 = np.array([-np.sqrt(3)/2.0, -0.5])
    n2 = np.array([np.sqrt(3)/2.0, -0.5])
    normals = np.array([n0, n1, n2])
    
    for centers in candidates:
        # Center the cluster
        centroid = np.mean(centers, axis=0)
        rel_centers = centers - centroid
        
        # Optimize rotation
        # Scan rotation from 0 to 60 degrees
        best_local_S = float('inf')
        best_local_rot = 0.0
        
        for rot_deg in np.arange(0, 60, 0.5):
            theta = np.deg2rad(rot_deg)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            
            # Rotate centers
            rotated_centers = rel_centers @ rot_mat.T
            
            # Calculate required inradius
            # For each normal, find max projection of any vertex
            # Vertices of unit hexagon
            # We need max over all vertices of all hexagons
            # This is equivalent to max over centers of (center . n + support(n))
            # But support(n) depends on the alignment of the inner hexagon relative to n.
            # Since we are rotating the whole cluster, the inner hexagons are also rotated.
            # So the support function of the inner hexagon in direction n is h(n - theta)?
            # No.
            # Let's just compute the max projection of the actual vertices.
            # For each center c_rot, the vertices are c_rot + hex_verts @ rot_mat.T
            # We need max | (c_rot + v_rot) . n |
            # This is max | c_rot . n + v_rot . n |
            # = | c_rot . n | + max | v_rot . n | ? No, max of sum is not sum of max.
            # But since we want the bounding box, we can just compute the max projection of the set of all vertices.
            
            # Efficient way:
            # The set of vertices is the Minkowski sum of the set of centers and the set of hex vertices.
            # The support function of a Minkowski sum is the sum of support functions.
            # h_{A+B}(u) = h_A(u) + h_B(u).
            # So, max_{p in Cluster} p . n = max_{c in Centers} c . n + max_{v in Hex} v . n.
            # Wait, this is for the support in direction n.
            # We need the width, or rather the max absolute projection.
            # The condition is | p . n | <= R_in.
            # This is equivalent to p . n <= R_in AND -p . n <= R_in.
            # So R_in >= max( max_p p . n, max_p -p . n ).
            # max_p p . n = h_Cluster(n).
            # max_p -p . n = h_Cluster(-n).
            # So R_in >= max( h_Cluster(n), h_Cluster(-n) ).
            
            # h_Cluster(n) = max_{c} c . n + h_Hex(n).
            # h_Hex(n) for a unit hexagon rotated by theta.
            # The vertices are hex_verts @ rot_mat.T.
            # h_Hex(n) = max_{v in hex_verts} (v @ rot_mat.T) . n = max_{v} v . (rot_mat @ n).
            # So we need to compute h_Hex for the rotated normals.
            
            # Let's precompute h_Hex for a unit hexagon for a given direction vector.
            def hex_support(direction):
                # direction is a unit vector
                # max dot product with vertices
                return np.max(hex_verts @ direction)
            
            # For this rotation theta:
            # Rotated normals: n_rot = normals @ rot_mat
            # Wait, if we rotate the object by theta, the support in direction n is h(n - theta) in original frame?
            # h_{Obj_rot}(n) = h_{Obj}(R^T n).
            # So we need h_Hex(R^T n).
            
            # Let's compute h_Hex for each normal direction transformed by inverse rotation.
            # inv_rot_mat = rot_mat.T
            # n_orig = normals @ inv_rot_mat
            
            # Calculate max projection of centers for each normal
            # proj_centers = rotated_centers @ normals.T
            # max_proj_centers = np.max(proj_centers, axis=0)
            
            # Calculate support of hex for each normal
            # n_orig = normals @ rot_mat.T
            # h_hex = np.array([hex_support(n) for n in n_orig])
            
            # Total support = max_proj_centers + h_hex
            # R_in_req = max( total_support, -min_proj_centers + h_hex(-n)? )
            # Actually, h_Cluster(-n) = max_c c . (-n) + h_Hex(-n)
            # = - min_c c . n + h_Hex(-n).
            # h_Hex(-n) = h_Hex(n) for centrally symmetric shapes? Yes, hexagon is symmetric.
            
            # So R_in_req = max( max_proj_centers + h_hex, -min_proj_centers + h_hex )
            # = max_proj_centers + h_hex if max_proj_centers >= -min_proj_centers?
            # No.
            # R_in >= max_proj_centers + h_hex
            # R_in >= -min_proj_centers + h_hex
            
            # So R_in = max( max_proj_centers + h_hex, -min_proj_centers + h_hex )
            
            min_proj_centers = np.min(proj_centers, axis=0)
            
            # Compute h_hex for the 3 normals
            # n_orig = normals @ rot_mat.T
            n_orig = normals @ rot_mat.T
            h_hex = np.array([hex_support(n) for n in n_orig])
            
            R_in_candidate = np.max(np.maximum(max_proj_centers + h_hex, -min_proj_centers + h_hex))
            
            S_candidate = (2.0 / np.sqrt(3.0)) * R_in_candidate
            
            if S_candidate < best_local_S:
                best_local_S = S_candidate
                best_local_rot = rot_deg

        if best_local_S < best_S:
            best_S = best_local_S
            best_config = rel_centers
            best_rotation = best_local_rot
            best_centers = centers # Store original to reconstruct if needed? No, we return relative + center.

    # Final Result
    # Add safety margin
    S_final = best_S + 1e-6
    
    # Construct inner list
    # We need to return the centers in the global coordinate system.
    # We centered them at origin. So global centers are best_config.
    # Wait, best_config is rel_centers.
    # The outer hexagon center is (0,0).
    
    # Convert to list of [x, y, angle_degrees]
    inner = []
    for c in best_config:
        inner.append([float(c[0]), float(c[1]), float(best_rotation)])
        
    # Outer hexagon: center (0,0), side S_final, rotation 0.0
    # Note: The rotation of the inner hexagons is best_rotation.
    # The outer hexagon rotation is 0.0.
    
    return inner, (0.0, 0.0), float(S_final), 0.0

# EVOLVE-BLOCK-END
