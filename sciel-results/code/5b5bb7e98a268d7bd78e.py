<<<<<<< SEARCH
def _compute_min_side_length(centers, angles, outer_angle_rad=0.0):
    """
    Compute the minimum side length S such that all inner hexagons fit inside
    the outer hexagon centered at origin with rotation outer_angle_rad.
    """
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    
    max_abs_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        for k in range(6):
            n = outer_normals[k]
            projs = verts @ n
            max_abs_proj = max(max_abs_proj, np.max(np.abs(projs)))
    
    S = max_abs_proj * 2.0 / math.sqrt(3.0)
    return S

def _point_in_polygon(p, poly):
    """Check if point p is inside convex polygon poly (6,2 array)."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        if cross < -1e-10:
            return False
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between segments p1-p2 and p3-p4."""
    # Check if segments intersect
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return 0.0
    
    # Check point-segment distances
    def pt_seg_dist(p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)
    
    d = min(pt_seg_dist(p1, p3, p4), pt_seg_dist(p2, p3, p4),
            pt_seg_dist(p3, p1, p2), pt_seg_dist(p4, p1, p2))
    return d

def _poly_separation(poly1, poly2):
    """
    Compute minimum distance between two convex polygons.
    Returns 0 if overlapping, positive if separated.
    """
    # Check if any vertex of one is inside the other
    for i in range(6):
        if _point_in_polygon(poly1[i], poly2):
            return 0.0
    for i in range(6):
        if _point_in_polygon(poly2[i], poly1):
            return 0.0
    
    # Check all edge pairs
    min_dist = float('inf')
    for i in range(6):
        for j in range(6):
            d = _seg_seg_dist(poly1[i], poly1[(i+1)%6], poly2[j], poly2[(j+1)%6])
            min_dist = min(min_dist, d)
    
    return min_dist

def _total_overlap_penalty(centers, angles, eps=0.01):
    """Compute total penalty for overlaps between inner hexagons."""
    polys = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _poly_separation(polys[i], polys[j])
            if d < eps:
                penalty += (eps - d) ** 2
    return penalty

def _objective(params, outer_angle_rad):
    """
    Objective function: minimize outer side length + overlap penalty.
    params: 3N array [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    """
    centers = np.column_stack([params[0::3], params[1::3]])
    angles = params[2::3]
    
    S = _compute_min_side_length(centers, angles, outer_angle_rad)
    pen = _total_overlap_penalty(centers, angles)
    
    return S + 100.0 * pen

def _is_valid(centers, angles, S, outer_angle_rad=0.0, tol=1e-4):
    """Check if configuration is valid: no overlaps, all inside outer hexagon."""
    polys = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = _poly_separation(polys[i], polys[j])
            if d < tol:
                return False
    
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    limit = S * math.sqrt(3) / 2.0
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        for k in range(6):
            n = outer_normals[k]
            projs = verts @ n
            if np.max(np.abs(projs)) > limit + tol:
                return False
    return True
=======
def _get_hex_vertices_vectorized(centers, angles):
    """
    Returns array of shape (N, 6, 2) with vertices for all hexagons.
    centers: (N, 2), angles: (N,)
    """
    N = len(centers)
    # Base vertices of unit hexagon
    base_theta = np.array([i * math.pi / 3.0 for i in range(6)])
    base_verts = np.column_stack([np.cos(base_theta), np.sin(base_theta)])
    
    # Rotation matrices for each hexagon
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    # (N, 2, 2) rotation matrices
    R = np.zeros((N, 2, 2))
    R[:, 0, 0] = cos_a
    R[:, 0, 1] = -sin_a
    R[:, 1, 0] = sin_a
    R[:, 1, 1] = cos_a
    
    # Rotate base vertices: (N, 6, 2)
    # base_verts is (6, 2). We want (N, 6, 2).
    # v_rot = R @ v_base
    # Using einsum: R is (N, 2, 2), v is (6, 2) -> result (N, 6, 2)
    # Actually, v_rot[i] = R[i] @ base_verts.T -> (2, 6) -> transpose to (6, 2)
    # Let's do it simpler:
    # v_rot = np.einsum('nij,jk->nik', R, base_verts.T).T -> (N, 6, 2)
    # Wait, R is (N, 2, 2). base_verts is (6, 2).
    # For each i, we want R[i] @ base_verts.T (2x6).
    # Result shape (N, 2, 6). Transpose last two dims -> (N, 6, 2).
    v_rot = np.einsum('nij,jk->nik', R, base_verts.T).transpose(0, 2, 1)
    
    # Translate
    v_final = v_rot + centers[:, np.newaxis, :]
    return v_final

def _sat_penalty_and_grad(centers, angles, eps=0.1):
    """
    Compute overlap penalty and gradient using SAT.
    Penalty is sum of squared penetrations for all pairs.
    """
    N = len(centers)
    verts = _get_hex_vertices_vectorized(centers, angles)
    
    penalty = 0.0
    grad = np.zeros_like(centers)
    grad_angles = np.zeros_like(angles)
    
    # Normals for each hexagon edge
    # Edge vectors are v[i+1] - v[i]. Normal is perpendicular.
    # Since hexagons are convex and regular, we can compute normals directly from angles.
    # Normal k for hex i is at angle angles[i] + k*pi/3.
    
    for i in range(N):
        for j in range(i + 1, N):
            # Check all 6 axes of i and 6 axes of j
            min_dist = float('inf')
            active_axis = None
            active_axis_idx = 0 # 0-5 for i's axes, 6-11 for j's axes
            
            # Axes from hex i
            for k in range(6):
                theta = angles[i] + k * math.pi / 3.0
                n = np.array([math.cos(theta), math.sin(theta)])
                
                # Project both polys onto n
                proj_i = verts[i] @ n
                proj_j = verts[j] @ n
                
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                # Separation
                if max_i < min_j:
                    dist = min_j - max_i
                elif max_j < min_i:
                    dist = min_i - max_j
                else:
                    dist = 0.0 # Overlap on this axis
                
                if dist < min_dist:
                    min_dist = dist
                    active_axis = n
                    active_axis_idx = k
            
            # Axes from hex j
            for k in range(6):
                theta = angles[j] + k * math.pi / 3.0
                n = np.array([math.cos(theta), math.sin(theta)])
                
                proj_i = verts[i] @ n
                proj_j = verts[j] @ n
                
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j:
                    dist = min_j - max_i
                elif max_j < min_i:
                    dist = min_i - max_j
                else:
                    dist = 0.0
                
                if dist < min_dist:
                    min_dist = dist
                    active_axis = n
                    active_axis_idx = 6 + k
            
            if min_dist < eps:
                # Penalty term: (eps - min_dist)^2
                # Gradient w.r.t positions:
                # d/dx (eps - d)^2 = -2(eps - d) * d(d)/dx
                # d = proj_j_max - proj_i_max (assuming this is the separating direction)
                # We need to know which vertices are the "touching" ones to compute gradient accurately.
                # For simplicity and robustness in gradient descent, we can use a smoothed approximation
                # or just push them apart along the normal.
                
                # Approximate gradient:
                # If i is to the left of j along n, increasing x_i along n decreases dist.
                # So grad_i = -2(eps-d) * n * (sign of separation)
                # Let's determine the sign.
                # We stored active_axis. We need to know if i is "before" j or vice versa.
                # Re-calculate projections on active_axis to determine direction.
                n = active_axis
                proj_i = verts[i] @ n
                proj_j = verts[j] @ n
                center_i_proj = np.mean(proj_i)
                center_j_proj = np.mean(proj_j)
                
                if center_i_proj < center_j_proj:
                    # i is "left", j is "right"
                    # dist = min_j - max_i
                    # d(dist)/d(centers_i) = -n (moving i right decreases dist)
                    # d(dist)/d(centers_j) = n (moving j right increases dist? No, min_j increases)
                    # Wait. If i is left, max_i is the rightmost of i. min_j is leftmost of j.
                    # dist = min_j - max_i.
                    # Increase x_i by n: max_i increases -> dist decreases. Grad_i = -n.
                    # Increase x_j by n: min_j increases -> dist increases. Grad_j = n.
                    sign_i = -1.0
                    sign_j = 1.0
                else:
                    sign_i = 1.0
                    sign_j = -1.0
                
                factor = -2.0 * (eps - min_dist)
                grad[i] += factor * sign_i * n
                grad[j] += factor * sign_j * n
                
                # Angle gradients are complex to compute analytically here.
                # We will rely on numerical differentiation for angles or ignore them initially
                # and let position optimization do the heavy lifting, then fine-tune angles.
                # However, the directive asks for gradient-based solver.
                # Let's use a numerical gradient for the whole objective in the optimizer.
                
                penalty += (eps - min_dist) ** 2
                
    return penalty, grad, grad_angles

def _compute_min_side_length(centers, angles, outer_angle_rad=0.0):
    """
    Compute the minimum side length S such that all inner hexagons fit inside
    the outer hexagon centered at origin with rotation outer_angle_rad.
    """
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    
    max_abs_proj = 0.0
    verts = _get_hex_vertices_vectorized(centers, angles)
    
    for k in range(6):
        n = outer_normals[k]
        # Project all vertices of all hexagons onto n
        # verts is (N, 6, 2). n is (2,)
        # projs is (N, 6)
        projs = np.einsum('ijk,j->ik', verts, n)
        max_abs_proj = max(max_abs_proj, np.max(np.abs(projs)))
    
    S = max_abs_proj * 2.0 / math.sqrt(3.0)
    return S

def _is_valid(centers, angles, S, outer_angle_rad=0.0, tol=1e-4):
    """Check if configuration is valid: no overlaps, all inside outer hexagon."""
    # Check overlaps
    N = len(centers)
    verts = _get_hex_vertices_vectorized(centers, angles)
    for i in range(N):
        for j in range(i + 1, N):
            # Simple SAT check for intersection
            separated = False
            # Axes from i
            for k in range(6):
                theta = angles[i] + k * math.pi / 3.0
                n = np.array([math.cos(theta), math.sin(theta)])
                proj_i = verts[i] @ n
                proj_j = verts[j] @ n
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    separated = True
                    break
            if separated:
                continue
            
            # Axes from j
            for k in range(6):
                theta = angles[j] + k * math.pi / 3.0
                n = np.array([math.cos(theta), math.sin(theta)])
                proj_i = verts[i] @ n
                proj_j = verts[j] @ n
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    separated = True
                    break
            
            if not separated:
                return False
                
    # Check inside outer
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    limit = S * math.sqrt(3) / 2.0
    for k in range(6):
        n = outer_normals[k]
        projs = np.einsum('ijk,j->ik', verts, n)
        if np.max(np.abs(projs)) > limit + tol:
            return False
            
    return True
>>>>>>> REPLACE

