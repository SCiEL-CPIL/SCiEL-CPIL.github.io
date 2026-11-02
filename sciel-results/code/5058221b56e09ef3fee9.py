import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 12 hexagon centers in a local coordinate system
    # Using a 7-hex flower plus 5 additional hexagons
    sqrt3 = np.sqrt(3)
    
    # Flower centers
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (-sqrt3, 0.0),
        (-sqrt3/2, -1.5),
        (sqrt3/2, -1.5)
    ]
    
    # Add 5 more from the next ring
    # Choosing these to create a somewhat compact, irregular shape
    centers.extend([
        (2*sqrt3, 0.0),
        (sqrt3/2, 3.0),
        (-sqrt3/2, 3.0),
        (-3*sqrt3/2, -1.5),
        (3*sqrt3/2, -1.5)
    ])
    
    centers = np.array(centers)
    
    # Define the 6 vertices of a unit regular hexagon (flat-top)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees?
    # Flat-top hexagon: vertices at 0, 60, ...
    # Side length 1. Radius 1.
    angles = np.deg2rad(np.arange(0, 360, 60))
    hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Generate all 72 vertices for the cluster
    # For each center, add the 6 vertices
    all_vertices = []
    for c in centers:
        all_vertices.append(c + hex_verts)
    all_vertices = np.vstack(all_vertices)
    
    def min_side_length(theta):
        # Rotate all vertices by theta
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        rotated_vertices = all_vertices @ rot_matrix.T
        
        # Normals for the 6 sides of the outer hexagon
        # Outer hexagon is fixed orientation (flat-top)
        # Normals are at 30, 90, 150, 210, 270, 330 degrees?
        # For a flat-top hexagon, the sides are vertical?
        # No, flat-top means the top side is horizontal.
        # Normal to top side is (0, 1).
        # Normals: 90, 150, 210, 270, 330, 30.
        normal_angles = np.deg2rad(np.arange(30, 360, 60))
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Project rotated vertices onto normals
        projections = rotated_vertices @ normals.T
        
        # For each normal, find min and max projection
        min_projs = np.min(projections, axis=0)
        max_projs = np.max(projections, axis=0)
        
        # Widths
        widths = max_projs - min_projs
        
        # The side length S is determined by the maximum width
        # Distance from center to side d = max(widths) / 2
        # S = d / (sqrt(3)/2) = max(widths) / sqrt(3)
        max_width = np.max(widths)
        side_length = max_width / sqrt3
        
        # Calculate the center of the outer hexagon
        # The center projection on each normal is the midpoint
        mid_projs = (min_projs + max_projs) / 2.0
        
        # We need to find (x, y) such that its projection on each normal is mid_projs
        # This is a linear system: normals.T @ center = mid_projs
        # Since there are 6 normals and 2 unknowns, it's overdetermined.
        # But it should be consistent.
        # We can use least squares or just pick 3 non-collinear normals.
        # Let's use the first 3 normals (30, 90, 150)
        A = normals[:3].T
        b = mid_projs[:3]
        
        # Solve A @ center = b
        # A is 2x3, b is 3x1.
        # Actually, A is 2x3 (2 rows, 3 cols).
        # We want center (2x1).
        # A.T @ center = b? No.
        # normals[i] dot center = mid_projs[i]
        # So normals.T (3x2) @ center (2x1) = mid_projs (3x1)
        center, residuals, rank, sv = np.linalg.lstsq(normals.T, mid_projs, rcond=None)
        
        return side_length, center

    # Search for optimal theta
    # Coarse search
    thetas = np.linspace(0, 2*np.pi, 361)
    best_theta = 0
    best_side = float('inf')
    best_center = (0,0)
    
    for t in thetas:
        side, center = min_side_length(t)
        if side < best_side:
            best_side = side
            best_theta = t
            best_center = center
            
    # Fine search (ternary search) around the best coarse theta
    # Interval width 2*pi/360
    delta = 2*np.pi/360
    a = best_theta - delta
    b = best_theta + delta
    
    # Ternary search
    for _ in range(50):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        f1, c1 = min_side_length(m1)
        f2, c2 = min_side_length(m2)
        
        if f1 < f2:
            b = m2
        else:
            a = m1
            
    theta_opt = (a + b) / 2
    side_opt, center_opt = min_side_length(theta_opt)
    
    # Construct the output
    # inner: rows (x, y, angle_degrees)
    # The angle of each inner hexagon is the global rotation theta_opt
    angle_deg = np.rad2deg(theta_opt)
    
    inner = []
    for c in centers:
        inner.append([c[0], c[1], angle_deg])
        
    # Outer hexagon center, side, angle
    # The outer hexagon angle is 0 (fixed in our calculation)
    # But we rotated the cluster.
    # The problem asks for "outer hexagon ... rotation".
    # We can return 0 for outer rotation, and the cluster rotation is handled by the inner hexagon angles?
    # No, the inner hexagons are part of the cluster.
    # If we rotate the cluster by theta, the inner hexagons are rotated by theta.
    # The outer hexagon is fixed at 0.
    # So return outer angle 0.
    
    return inner, (float(center_opt[0]), float(center_opt[1])), float(side_opt), 0.0
# EVOLVE-BLOCK-END
