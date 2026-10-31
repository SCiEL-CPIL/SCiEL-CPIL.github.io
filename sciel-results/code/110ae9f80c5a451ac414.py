import numpy as np
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate axial coordinates for hexagons
    # Ring 0
    ring0 = [(0, 0)]
    # Ring 1
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    # Ring 2
    ring2 = [
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -2), (-2, -1),
        (-2, 0), (-2, 1), (-1, 2), (0, 2), (1, 2), (2, 1)
    ]
    
    base_hexes = ring0 + ring1
    
    # Function to get vertices of a unit hexagon at axial (q, r) with angle 0 (flat-top)
    # Flat-top unit hex vertices relative to center:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    def axial_to_pixel(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        return x, y

    def get_vertices(q, r):
        cx, cy = axial_to_pixel(q, r)
        verts = []
        for i in range(6):
            angle = np.deg2rad(60 * i)
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            verts.append((vx, vy))
        return verts

    # Precompute vertices for base hexes
    base_verts = []
    for q, r in base_hexes:
        base_verts.extend(get_vertices(q, r))
    
    # Precompute vertices for ring2 hexes
    ring2_verts = []
    for q, r in ring2:
        ring2_verts.append(get_vertices(q, r))

    best_S = float('inf')
    best_config = None
    best_phi = 0.0
    
    # Try all combinations of 4 hexagons from ring2
    for combo in itertools.combinations(range(12), 4):
        # Get vertices for this combo
        current_verts = list(base_verts)
        for idx in combo:
            current_verts.extend(ring2_verts[idx])
        
        # Convert to numpy array for speed
        verts = np.array(current_verts)
        
        # Optimize rotation phi
        # Symmetry period is pi/3 (60 deg)
        # We can sample phi or use optimization.
        # The function S(phi) is the max of linear functions, so it's convex-ish?
        # Let's just sample finely or use scipy minimize.
        # Given 120s limit, sampling 100 points per combo * 495 combos = 49500 evals.
        # Each eval is O(N_verts) ~ 66 points. Very fast.
        
        min_S_for_combo = float('inf')
        best_phi_for_combo = 0.0
        
        # Sample phi from 0 to 60 degrees
        # Step 0.5 degrees
        for phi_deg in np.arange(0, 60, 0.5):
            phi = np.deg2rad(phi_deg)
            c, s = np.cos(-phi), np.sin(-phi)
            rot_mat = np.array([[c, -s], [s, c]])
            
            # Rotate vertices
            rotated_verts = verts @ rot_mat.T
            
            # Calculate required S for each vertex
            # Constraints:
            # |x| <= S
            # |y| <= S * sqrt(3)/2
            # |sqrt(3)x + y| <= S * sqrt(3)
            # |sqrt(3)x - y| <= S * sqrt(3)
            
            x = rotated_verts[:, 0]
            y = rotated_verts[:, 1]
            
            S1 = np.abs(x)
            S2 = np.abs(y) * (2.0 / np.sqrt(3))
            S3 = np.abs(np.sqrt(3) * x + y) / np.sqrt(3)
            S4 = np.abs(np.sqrt(3) * x - y) / np.sqrt(3)
            
            S_max = np.max(np.maximum(np.maximum(S1, S2), np.maximum(S3, S4)))
            
            if S_max < min_S_for_combo:
                min_S_for_combo = S_max
                best_phi_for_combo = phi_deg
                
        if min_S_for_combo < best_S:
            best_S = min_S_for_combo
            best_phi = best_phi_for_combo
            best_config = [base_hexes] + [ring2[i] for i in combo]
            
    # Construct the inner list
    # Format: (x, y, angle_degrees)
    # The angle of the inner hexagons is 0 relative to their local frame.
    # The container is rotated by best_phi.
    # The return value asks for inner rows (x, y, angle_degrees).
    # The angle_degrees is the rotation of the inner hexagon.
    # If we rotated the container, it's equivalent to rotating the inner hexagons by -best_phi.
    # So inner angle = -best_phi.
    
    inner = []
    for q, r in best_config:
        x, y = axial_to_pixel(q, r)
        inner.append([x, y, -best_phi])
        
    return inner, (0.0, 0.0), best_S, best_phi

# EVOLVE-BLOCK-END
