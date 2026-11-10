<<<<<<< SEARCH
def _hybrid_optimize(pts, time_budget):
    """Combine repulsive force and SOCP-style subgradient optimization."""
    pts = _normalize(pts).copy()
    start = time.time()
    best_r = _covering_radius_fast(pts)
    best_pts = pts.copy()
    
    # Phase 1: Repulsive force to spread points
    phase1_budget = min(time_budget * 0.4, 30.0)
    pts, r1 = _repulsive_force_optimize(pts, phase1_budget)
    if r1 < best_r:
        best_r = r1
        best_pts = pts.copy()
    
    # Phase 2: SOCP-style subgradient to target worst regions
    remaining = time_budget - (time.time() - start)
    if remaining > 5.0:
        pts, r2 = _socp_covering_optimize(pts, remaining, n_grid=80)
        if r2 < best_r:
            best_r = r2
            best_pts = pts.copy()
    
    # Phase 3: Fine-tune with local coordinate descent
    remaining = time.time() - start
    if remaining > 10.0:
        pts, r3 = _local_optimize_fast(best_pts, remaining)
        if r3 < best_r:
            best_r = r3
            best_pts = pts.copy()
    
    return best_pts, best_r
=======
def _facet_circumradius(a, b, c):
    """
    Compute the angular circumradius of a triangle on the unit sphere.
    Returns the radius (radians) and the circumcenter (unit vector).
    """
    # Normal vector
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return 0.0, a
    
    n = n / nn
    
    # The circumcenter lies on the normal line through the centroid? No.
    # It is the intersection of the normal line and the plane?
    # For a triangle on a sphere, the circumcenter is the point on the sphere
    # such that its distance to A, B, C is equal.
    # This point is along the normal vector n.
    # Let O = k * n.
    # |O - A|^2 = |O - B|^2
    # |O|^2 - 2 O.A + |A|^2 = |O|^2 - 2 O.B + |B|^2
    # Since |A|=|B|=1, 2 O.(B-A) = 0 -> O is orthogonal to B-A.
    # Similarly O is orthogonal to C-A.
    # So O is parallel to n.
    # Let O = c * n.
    # We want to minimize |O - A|, or rather find the O on the line k*n
    # that is equidistant.
    # Actually, the circumcenter of the spherical triangle is the point P
    # such that P.A = P.B = P.C = cos(R).
    # P is along the normal n?
    # If P = c*n, then P.A = c (n.A).
    # We need c (n.A) = c (n.B) = c (n.C).
    # This implies n.A = n.B = n.C.
    # Is this true?
    # n = (b-a) x (c-a).
    # n.A = A . ((b-a) x (c-a)) = (A x (b-a)) . (c-a) ...
    # Scalar triple product A.(b-a, c-a) = A.(b,c) - A.(a,c) ...
    # A.(b x c - a x c - b x a + a x a) = A.(b x c) - A.(a x c) - A.(b x a)
    # = A.(b x c) - A.(a x c) + A.(a x b)
    # B.(b x c) = 0.
    # So n.A = A.(b x c) + A.(a x b).
    # This is not necessarily equal for A, B, C.
    # Wait, the circumcenter of a spherical triangle is NOT generally on the normal of the planar triangle.
    # It is the pole of the great circle? No.
    
    # Correct method:
    # The circumcenter P satisfies P.A = P.B = P.C.
    # P is a linear combination of A, B, C.
    # P = x A + y B + z C.
    # P.A = x + y (A.B) + z (A.C) = cos R
    # P.B = x (A.B) + y + z (B.C) = cos R
    # P.C = x (A.C) + y (B.C) + z = cos R
    # Also |P| = 1.
    
    # Let G be the Gram matrix [[1, c_ab, c_ac], [c_ab, 1, c_bc], [c_ac, c_bc, 1]].
    # Let v = [x, y, z]^T.
    # The equations are G v = [cos R, cos R, cos R]^T? No.
    # P.A = v^T G e_1.
    # We want P.A = P.B = P.C = c.
    # So G v = c * [1, 1, 1]^T.
    # v = c * G^{-1} [1, 1, 1]^T.
    # Let u = G^{-1} [1, 1, 1]^T.
    # P = c * u.
    # |P|^2 = 1 => c^2 u^T u = 1 => c = 1 / ||u||.
    # Then R = arccos(c).
    
    c_ab = np.dot(a, b)
    c_ac = np.dot(a, c)
    c_bc = np.dot(b, c)
    
    G = np.array([
        [1.0, c_ab, c_ac],
        [c_ab, 1.0, c_bc],
        [c_ac, c_bc, 1.0]
    ])
    
    try:
        inv_G = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return 0.0, a
        
    ones = np.array([1.0, 1.0, 1.0])
    u = inv_G @ ones
    
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-12:
        return 0.0, a
        
    c_val = 1.0 / norm_u
    c_val = np.clip(c_val, -1.0, 1.0)
    
    P = u * c_val
    P = P / np.linalg.norm(P)
    
    R = np.arccos(c_val)
    
    return R, P


def _facet_targeted_optimize(pts, time_budget):
    """
    Optimize by targeting the worst facet (largest circumradius).
    Move the vertices of the worst facet towards its circumcenter.
    """
    pts = _normalize(pts).copy()
    start = time.time()
    best_r = _covering_radius_hull(pts)
    best_pts = pts.copy()
    
    step_size = 0.05
    
    while time.time() - start < time_budget:
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        worst_r = -1.0
        worst_f = None
        worst_P = None
        
        for f in hull.simplices:
            tri = pts[f]
            r, P = _facet_circumradius(tri[0], tri[1], tri[2])
            if r > worst_r:
                worst_r = r
                worst_f = f
                worst_P = P
                
        if worst_f is None:
            break
            
        # Check if we improved
        if worst_r < best_r - 1e-12:
            best_r = worst_r
            best_pts = pts.copy()
            
        # If the worst radius is very small, we are done
        if worst_r < 1e-6:
            break
            
        # Move the vertices of the worst facet towards P
        # Gradient direction for vertex i:
        # We want to increase the dot product P . pts[i]
        # So move pts[i] in the direction of P (projected onto tangent plane)
        
        for idx in worst_f:
            p_i = pts[idx]
            # Direction towards P
            d = worst_P - p_i
            # Project to tangent plane at p_i
            d_tan = d - np.dot(d, p_i) * p_i
            norm_d = np.linalg.norm(d_tan)
            if norm_d < 1e-12:
                continue
            d_tan = d_tan / norm_d
            
            # Move
            pts[idx] = pts[idx] + step_size * d_tan
            pts[idx] = pts[idx] / np.linalg.norm(pts[idx])
            
        # Decay step size
        step_size *= 0.95
        if step_size < 1e-6:
            step_size = 0.01 # Restart with larger step? Or just break?
            # Let's try restarting step size to escape local minima
            # But be careful not to oscillate.
            # For now, just keep decaying.
            
    return best_pts, best_r
>>>>>>> REPLACE

Now update `construct` to use this new optimizer.

