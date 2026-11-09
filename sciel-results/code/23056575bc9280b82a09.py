import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define centers for 11 unit hexagons
    # 1 center
    centers = [(0.0, 0.0)]
    
    # 6 neighbors in ring 1
    # Distance sqrt(3)
    d1 = np.sqrt(3.0)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        x = d1 * np.cos(angle)
        y = d1 * np.sin(angle)
        centers.append((x, y))
        
    # 4 hexagons in ring 2 (corners)
    # Distance 3
    d2 = 3.0
    # Angles 30, 90, 150, 270 degrees
    angles_ring2 = [30, 90, 150, 270]
    for deg in angles_ring2:
        angle = np.deg2rad(deg)
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        centers.append((x, y))
        
    # Total 11 centers
    # Inner hexagons are aligned with angle 0
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    # Get all vertices of the inner hexagons
    # Unit hexagon vertices (angle 0, side 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Distance from center to vertex is 1
    hex_verts = []
    for i in range(6):
        angle = np.deg2rad(60 * i)
        hex_verts.append((np.cos(angle), np.sin(angle)))
    hex_verts = np.array(hex_verts)
    
    all_points = []
    for c in centers:
        # Translate vertices
        pts = c[0] + hex_verts[:, 0]
        pts_y = c[1] + hex_verts[:, 1]
        for i in range(6):
            all_points.append((pts[i], pts_y[i]))
            
    all_points = np.array(all_points)
    
    # Optimize outer hexagon rotation
    # Outer hexagon side S, apothem a = S * sqrt(3) / 2
    # Normals of outer hexagon sides are at phi + 30 + 60*k
    # Point P is inside if dot(P, n_k) <= a for all k
    
    best_S = float('inf')
    best_phi = 0.0
    
    # Scan rotation phi from 0 to 60 degrees
    # Step 0.1 deg for precision
    for phi_deg in np.arange(0, 60.1, 0.1):
        phi = np.deg2rad(phi_deg)
        # Normals
        normals = []
        for k in range(6):
            angle = phi + np.deg2rad(30 + 60 * k)
            normals.append((np.cos(angle), np.sin(angle)))
        normals = np.array(normals)
        
        # Project all points onto normals
        # proj[i, j] = dot(all_points[i], normals[j])
        projs = all_points @ normals.T
        
        # Max projection for each normal
        max_projs = np.max(projs, axis=0)
        
        # Required apothem is max of these
        a_required = np.max(max_projs)
        
        # Calculate S
        S = a_required * 2.0 / np.sqrt(3.0)
        
        if S < best_S:
            best_S = S
            best_phi = phi_deg
            
    # Return inner, center, side, angle
    # Center is (0,0)
    # Side is best_S
    # Angle is best_phi
    return inner, (0.0, 0.0), best_S, best_phi
# EVOLVE-BLOCK-END
