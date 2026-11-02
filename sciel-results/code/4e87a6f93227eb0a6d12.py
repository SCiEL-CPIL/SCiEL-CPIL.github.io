import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define lattice vectors
    e1 = np.array([np.sqrt(3), 0.0])
    e2 = np.array([np.sqrt(3)/2.0, 1.5])
    
    # 7-cluster centers
    centers = [
        np.array([0.0, 0.0]),
        e1,
        e2,
        e2 - e1,
        -e1,
        -e2,
        e1 - e2
    ]
    
    # Add 5 more from the next ring
    # Next ring points (distance 2)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,-1 is already there? No (1,-1) is dist 1)
    # Let's add 5 specific ones to make it compact
    # We'll add the ones that extend the "flat" sides of the 7-cluster
    # The 7-cluster has vertices at angles 0, 60, 120, 180, 240, 300.
    # Adding points at the midpoints of the edges of the bounding hexagon?
    
    # Let's just add 5 points:
    c8 = 2*e1
    c9 = e1 + e2
    c10 = 2*e2
    c11 = -e1 + 2*e2
    c12 = -2*e1 + e2
    
    centers.extend([c8, c9, c10, c11, c12])
    
    # Convert to list of arrays
    centers = np.array(centers)
    
    # Unit hexagon vertices (relative to center)
    # Side length 1
    # Vertices at angles 0, 60, ...
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2.0],
        [-0.5, np.sqrt(3)/2.0],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2.0],
        [0.5, -np.sqrt(3)/2.0]
    ])
    
    # Function to calculate minimal S for a given rotation theta
    def calc_S(theta):
        # Rotate centers
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        rot_centers = centers @ R.T
        
        # Rotate hex vertices
        rot_verts = hex_verts @ R.T
        
        # For each inner hexagon, get all vertices
        # all_points = rot_centers[:, np.newaxis, :] + rot_verts[np.newaxis, :, :]
        
        # We need to find the max "hexagonal norm" of these points
        # Outer hexagon is fixed orientation (flat-top? or pointy-top?)
        # Let's assume outer hexagon has vertices at 0, 60, ...
        # Constraints: |x| <= S, |x/2 + y*sqrt(3)/2| <= S*sqrt(3)/2, |-x/2 + y*sqrt(3)/2| <= S*sqrt(3)/2
        # This corresponds to a hexagon with vertices at 0, 60, ...
        # Apothem A = S * sqrt(3)/2
        # Norm A(p) = max( |p_x|, |p_x/2 + p_y*sqrt(3)/2|, |-p_x/2 + p_y*sqrt(3)/2| )
        # S = 2/sqrt(3) * A(p)
        
        max_A = 0.0
        for i in range(N):
            pts = rot_centers[i] + rot_verts
            # Calculate A for each point
            # A = max(|x|, |x/2 + y*sqrt(3)/2|, |-x/2 + y*sqrt(3)/2|)
            abs_x = np.abs(pts[:, 0])
            abs_term2 = np.abs(pts[:, 0]/2.0 + pts[:, 1]*np.sqrt(3)/2.0)
            abs_term3 = np.abs(-pts[:, 0]/2.0 + pts[:, 1]*np.sqrt(3)/2.0)
            
            A_vals = np.maximum(abs_x, np.maximum(abs_term2, abs_term3))
            max_A = max(max_A, np.max(A_vals))
            
        return (2.0 / np.sqrt(3.0)) * max_A

    # Optimize theta
    # The function is periodic with period pi/3 (60 degrees) due to hexagon symmetry
    # We search in [0, pi/6] (30 degrees) due to symmetry of the norm?
    # Actually, the cluster is not symmetric, so we search [0, pi/3)
    
    best_S = float('inf')
    best_theta = 0.0
    
    # Grid search + refinement
    for theta in np.linspace(0, np.pi/3.0, 1000):
        S = calc_S(theta)
        if S < best_S:
            best_S = S
            best_theta = theta
            
    # Refine
    # Ternary search around best_theta
    lo = max(0, best_theta - 0.01)
    hi = min(np.pi/3.0, best_theta + 0.01)
    
    for _ in range(50):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if calc_S(m1) < calc_S(m2):
            hi = m2
        else:
            lo = m1
        if hi - lo < 1e-6:
            break
            
    best_theta = (lo + hi) / 2.0
    best_S = calc_S(best_theta)
    
    # Construct final output
    c = np.cos(best_theta)
    s = np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    
    final_centers = centers @ R.T
    final_verts = hex_verts @ R.T # Not needed for output, just centers and angle
    
    inner = []
    for i in range(N):
        x, y = final_centers[i]
        # Angle of inner hexagon
        # The inner hexagon is rotated by best_theta
        # The output format is (x, y, angle_degrees)
        # The angle is the rotation of the inner hexagon relative to the global frame
        # Since we rotated the cluster by best_theta, the hexagons are rotated by best_theta
        angle_deg = np.degrees(best_theta)
        inner.append([float(x), float(y), angle_deg])
        
    return inner, (0.0, 0.0), float(best_S), 0.0
# EVOLVE-BLOCK-END
