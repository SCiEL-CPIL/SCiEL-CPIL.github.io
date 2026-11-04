<<<<<<< SEARCH
def _compute_enclosing_hexagon(vertices, outer_angle_deg):
    """
    Given a set of 2D points (vertices of all inner hexagons) and the outer
    hexagon rotation angle, compute the minimum side length and center of a
    regular hexagon with that rotation that contains all points.
    
    Returns (center_x, center_y, side_length).
    """
    normals = _outer_hex_normals(outer_angle_deg)
    # normals shape: (6, 2)
    # For each normal direction, compute min and max dot product
    # points shape: (M, 2)
    dots = vertices @ normals.T  # (M, 6)
    mins = dots.min(axis=0)  # (6,)
    maxs = dots.max(axis=0)  # (6,)
    ranges = maxs - mins  # (6,)
    
    # The apothem must be at least half the maximum range
    apothem = np.max(ranges) / 2.0
    side_length = 2.0 * apothem / math.sqrt(3)
    
    # The center in each normal direction is the midpoint
    # center_dot_k = (mins[k] + maxs[k]) / 2
    # We need to find (cx, cy) such that for each k:
    #   dot((cx, cy), normal_k) = center_dot_k
    # This is an overdetermined system; use least squares.
    center_dots = (mins + maxs) / 2.0  # (6,)
    # Solve: normals @ (cx, cy).T = center_dots
    # normals is (6, 2), so we solve the 2x2 normal equations
    A = normals.T  # (2, 6)
    # Least squares: (A^T A) c = A^T b
    ATA = A.T @ A  # (2, 2)
    ATb = A.T @ center_dots  # (2,)
    try:
        center = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        center = np.array([0.0, 0.0])
    
    return center[0], center[1], side_length

def _objective(positions, angles, outer_angle_deg):
    """
    Compute the outer hexagon side length for given inner hexagon positions
    and angles.
    """
    all_verts = []
    for i in range(N):
        cx, cy = positions[i]
        ang = angles[i]
        verts = _hex_vertices(cx, cy, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (6*N, 2)
    
    cx, cy, L = _compute_enclosing_hexagon(all_verts, outer_angle_deg)
    return L
=======
def _objective(params):
    """
    Compute the outer hexagon side length for a given set of parameters.
    params: 
      0..21: x, y for 11 hexagons
      22..32: angles for 11 hexagons
      33: outer angle
      34: outer center x
      35: outer center y
    """
    positions = params[:22].reshape(N, 2)
    angles = params[22:33]
    outer_angle_deg = params[33]
    outer_center = params[34:36]
    
    # Compute all vertices
    all_verts = np.empty((N * 6, 2))
    for i in range(N):
        cx, cy = positions[i]
        ang = angles[i]
        verts = _hex_vertices(cx, cy, ang)
        all_verts[i*6:(i+1)*6] = verts
    
    # Compute normals
    normals = _outer_hex_normals(outer_angle_deg) # (6, 2)
    
    # Calculate max distance from outer center to any vertex in any normal direction
    # dist_k = max_v | dot(v - c, n_k) |
    # We want min max_k dist_k
    
    # Shift vertices by center
    shifted = all_verts - outer_center # (M, 2)
    
    # Project onto normals
    # dots shape (M, 6)
    dots = shifted @ normals.T
    
    # For each normal direction k, find the max absolute distance
    max_abs_dots = np.max(np.abs(dots), axis=0) # (6,)
    
    # The required apothem is the max of these
    max_apothem = np.max(max_abs_dots)
    
    # Side length
    side_length = 2.0 * max_apothem / math.sqrt(3)
    
    return side_length
>>>>>>> REPLACE

