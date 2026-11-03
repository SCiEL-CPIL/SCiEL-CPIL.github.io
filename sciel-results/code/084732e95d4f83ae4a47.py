import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 3x4 grid of hexagons
    # Using axial coordinates
    centers = []
    for r in range(3):
        for q in range(4):
            # Convert axial to cartesian
            x = (np.sqrt(3) / 2) * (q + r / 2.0)
            y = (3 / 2.0) * r
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the grid
    mean_pos = np.mean(centers, axis=0)
    centers = centers - mean_pos
    
    # Determine optimal rotation
    # We want to minimize the max distance to the boundary of the outer hexagon
    # Boundary distance in direction theta for side S:
    # D(theta) = S * (sqrt(3)/2) / cos((theta - 30) % 60 - 30)  ... roughly
    # Let's just try a few rotations and pick the best S
    
    best_S = float('inf')
    best_rotation = 0.0
    
    # Try rotations from 0 to 30 degrees
    for rot_deg in np.linspace(0, 30, 31):
        theta = np.radians(rot_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_centers = centers @ R.T
        
        # Calculate required S for this rotation
        max_req_S = 0.0
        for cx, cy in rotated_centers:
            dist = np.sqrt(cx**2 + cy**2)
            angle = np.arctan2(cy, cx)
            
            # Normalize angle to 0-60
            ang_deg = np.degrees(angle) % 60
            
            # Distance to boundary of outer hex (side S) in this direction
            # Apothem is at 30 deg. Vertex at 0 deg.
            # dist_boundary = S * (sqrt(3)/2) / cos((ang_deg - 30))
            # Inner hex radius in this direction (aligned with outer? No, inner hexes are rotated by rot_deg too)
            # Wait, if we rotate the grid, we should rotate the hexes too to keep them aligned?
            # Yes, usually optimal to align inner hexes with each other.
            # So inner hex angle is rot_deg.
            # Inner hex radius in direction (angle - rot_deg)
            inner_ang = (ang_deg - rot_deg) % 60
            # Inner hex apothem at 30, vertex at 0
            inner_r = (np.sqrt(3)/2) / np.cos(np.radians(inner_ang - 30))
            
            # Constraint: dist + inner_r <= dist_boundary
            # dist + inner_r <= S * (sqrt(3)/2) / cos(np.radians(ang_deg - 30))
            # S >= (dist + inner_r) * cos(np.radians(ang_deg - 30)) * 2 / sqrt(3)
            
            req_S = (dist + inner_r) * np.cos(np.radians(ang_deg - 30)) * 2 / np.sqrt(3)
            if req_S > max_req_S:
                max_req_S = req_S
        
        if max_req_S < best_S:
            best_S = max_req_S
            best_rotation = rot_deg
            
    # Apply best rotation
    theta = np.radians(best_rotation)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    final_centers = centers @ R.T
    
    # Construct inner list
    inner = []
    for cx, cy in final_centers:
        inner.append([float(cx), float(cy), float(best_rotation)])
        
    return inner, (0.0, 0.0), float(best_S), 0.0
# EVOLVE-BLOCK-END
