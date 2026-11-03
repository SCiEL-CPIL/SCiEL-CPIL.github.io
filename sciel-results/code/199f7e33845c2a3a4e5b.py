def _min_enclosing_hex(verts, outer_angle_deg):
    """Find min enclosing hexagon for given vertices and orientation.
    Returns (cx, cy, side) or None."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    nx = np.cos(angles)
    ny = np.sin(angles)
    normals = np.column_stack([nx, ny]) # (6, 2)
    
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0
        
    # Project vertices onto each normal
    # dots[i, j] = v_i . n_j
    dots = verts @ normals.T # (M, 6)
    
    H = np.max(dots, axis=0) # (6,) max projection for each normal
    L = np.min(dots, axis=0) # (6,) min projection for each normal
    
    def cost(c):
        # c is (2,)
        # n_j . c
        nc = normals @ c # (6,)
        # H_j - n_j . c
        upper = H - nc
        # -L_j + n_j . c
        lower = -L + nc
        # Max of these for each j, then max over j
        return np.max(np.maximum(upper, lower))
        
    # Initial guess: centroid
    c0 = np.mean(verts, axis=0)
    
    try:
        res = minimize(cost, c0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000})
        if res.success:
            cx, cy = res.x
            R = res.fun
            side = R * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    return None
