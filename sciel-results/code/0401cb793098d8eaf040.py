import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate 11 hexagon centers
    # 1 at (0,0)
    # 6 in ring 1 (distance sqrt(3))
    # 4 in ring 2 (distance 2*sqrt(3))
    
    centers = []
    centers.append((0.0, 0.0))
    
    # Ring 1
    for i in range(6):
        angle = np.deg2rad(i * 60)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2: Need 4 more.
    # Ring 2 has 12 positions at angles 0, 30, 60, ...
    # To balance, pick 0, 90 (closest), 180, 270 (closest).
    # Closest to 90 is 60 or 120? 
    # Let's pick 0, 60, 180, 240 to start? Or 0, 120, 180, 300?
    # Let's try to pick 4 points that are roughly 90 deg apart.
    # Angles available: 0, 30, 60, 90(no), 120, 150, 180, 210, 240, 270(no), 300, 330.
    # Closest to 90: 60 (diff 30) or 120 (diff 30).
    # Closest to 270: 240 (diff 30) or 300 (diff 30).
    # Let's pick 0, 120, 180, 300.
    # 0 and 180 are opposite. 120 and 300 are opposite.
    # This creates a symmetric cross-like shape (rotated).
    
    ring2_angles = [0, 120, 180, 300]
    for angle_deg in ring2_angles:
        angle = np.deg2rad(angle_deg)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
        
    # Total 11 centers
    
    # Calculate minimal enclosing hexagon
    # Vertices of a unit hexagon (side 1) aligned with x-axis:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    min_S = float('inf')
    
    # Search outer rotation
    # The optimal rotation is likely aligned with the cluster symmetry.
    # Symmetry of our cluster (0, 120, 180, 300 + ring 1) is 180 deg?
    # Ring 1 has 6-fold symmetry.
    # The added 4 break 6-fold to 2-fold (or 1-fold).
    # Let's search 0 to 60 deg with fine steps.
    
    for phi_deg in np.linspace(0, 60, 601):
        phi = np.deg2rad(phi_deg)
        
        # Normals of the outer hexagon sides
        # Side normals are at angles phi, phi+60, ...
        # Distance from center to side is A = S * sqrt(3)/2
        # Condition: V . n <= A for all vertices V and normals n
        
        max_proj = 0.0
        
        for cx, cy in centers:
            # Translate vertices to center
            verts = hex_verts + np.array([cx, cy])
            
            for k in range(6):
                angle_n = phi + np.deg2rad(k * 60)
                n = np.array([np.cos(angle_n), np.sin(angle_n)])
                
                # Project all vertices onto normal
                projs = verts @ n
                max_proj = max(max_proj, np.max(projs))
                
        # S * sqrt(3)/2 >= max_proj
        S = 2.0 / np.sqrt(3.0) * max_proj
        if S < min_S:
            min_S = S
            
    # The result is the side length
    # We return the centers, outer center (0,0), side length, and optimal angle
    # We need to store the optimal angle found
    
    # Re-run to get optimal angle
    best_phi = 0
    min_S = float('inf')
    for phi_deg in np.linspace(0, 60, 601):
        phi = np.deg2rad(phi_deg)
        max_proj = 0.0
        for cx, cy in centers:
            verts = hex_verts + np.array([cx, cy])
            for k in range(6):
                angle_n = phi + np.deg2rad(k * 60)
                n = np.array([np.cos(angle_n), np.sin(angle_n)])
                projs = verts @ n
                max_proj = max(max_proj, np.max(projs))
        S = 2.0 / np.sqrt(3.0) * max_proj
        if S < min_S:
            min_S = S
            best_phi = phi_deg
            
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    return inner, (0.0, 0.0), min_S, best_phi
# EVOLVE-BLOCK-END
