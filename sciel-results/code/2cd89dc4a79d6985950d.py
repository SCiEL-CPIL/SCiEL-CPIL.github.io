import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Define the 1-2-3-3-2-1 cluster of 12 hexagons
    # This is a compact hexagonal arrangement
    # Centers are defined relative to the center of the cluster
    # dx = sqrt(3), dy = 1.5
    dx = np.sqrt(3)
    dy = 1.5
    
    centers = []
    # Row 0: 1 hex at (0, 0)
    centers.append((0, 0))
    
    # Row 1: 2 hexes at y = dy
    centers.append((-dx/2, dy))
    centers.append((dx/2, dy))
    
    # Row 2: 3 hexes at y = 2*dy
    centers.append((-dx, 2*dy))
    centers.append((0, 2*dy))
    centers.append((dx, 2*dy))
    
    # Row 3: 3 hexes at y = 3*dy
    centers.append((-dx, 3*dy))
    centers.append((0, 3*dy))
    centers.append((dx, 3*dy))
    
    # Row 4: 2 hexes at y = 4*dy
    centers.append((-dx/2, 4*dy))
    centers.append((dx/2, 4*dy))
    
    # Row 5: 1 hex at y = 5*dy
    centers.append((0, 5*dy))
    
    # The cluster is not centered at origin.
    # Calculate center of mass
    com_y = np.mean([c[1] for c in centers])
    com_x = np.mean([c[0] for c in centers])
    
    # Shift centers to center of mass
    centers = [(x - com_x, y - com_y) for x, y in centers]
    
    # Generate vertices for each unit hexagon
    # Unit hexagon vertices relative to center
    hex_verts_rel = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # List of all vertices for all hexagons
    all_vertices = []
    for cx, cy in centers:
        verts = hex_verts_rel + np.array([cx, cy])
        all_vertices.extend(verts.tolist())
        
    all_vertices = np.array(all_vertices)
    
    # Optimize the orientation of the outer hexagon
    # The outer hexagon has side length s.
    # Its normals are at angles theta + 90, theta + 210, theta + 330 (degrees)
    # Or simply, the constraints are on the projections onto the 3 axes of symmetry.
    # Let the outer hexagon have angle alpha.
    # The "width" in direction phi is 2 * s * cos(phi - alpha) ? No.
    # The distance from center to side in direction phi is s * sqrt(3)/2 * |cos(phi - alpha)|?
    # Actually, for a regular hexagon, the support function h(u) = s * max(0, cos(...))?
    # Simpler: The condition that a point p is inside a regular hexagon of side s centered at origin with angle alpha is:
    # |p dot n_k| <= s * sqrt(3)/2 for k=0,1,2 where n_k are the normals.
    
    # We want to minimize s such that for all vertices v:
    # |v dot n_k(theta)| <= s * sqrt(3)/2 for all k.
    # So s >= max_k |v dot n_k(theta)| / (sqrt(3)/2) for all v.
    # s(theta) = max_v ( max_k |v dot n_k(theta)| ) / (sqrt(3)/2)
    
    best_s = float('inf')
    best_theta = 0
    
    # Scan theta from 0 to 60 degrees
    for theta_deg in np.linspace(0, 60, 600):
        theta = np.deg2rad(theta_deg)
        # Normals of the outer hexagon sides
        # If hexagon angle is theta, vertices are at theta + 60*i.
        # Sides are perpendicular to vectors at theta + 30 + 60*i.
        # So normals are at angles theta + 30, theta + 90, theta + 150.
        normals = []
        for i in range(3):
            angle = theta + np.deg2rad(30 + 60*i)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0
        for v in all_vertices:
            for n in normals:
                proj = abs(np.dot(v, n))
                if proj > max_proj:
                    max_proj = proj
        
        s = max_proj / (np.sqrt(3)/2)
        if s < best_s:
            best_s = s
            best_theta = theta_deg
            
    # Construct the result
    inner_hex_data = np.array([[x, y, 0] for x, y in centers])
    outer_hex_data = np.array([0, 0, best_theta])
    
    return inner_hex_data, outer_hex_data, best_s
