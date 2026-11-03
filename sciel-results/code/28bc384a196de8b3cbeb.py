def objective(params):
    n = N
    # Extract positions
    positions = np.array(params[:2 * n]).reshape(n, 2)
    cx = params[2 * n]
    cy = params[2 * n + 1]
    theta = params[2 * n + 2]
    
    # Calculate side length
    outer_norm_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])) + np.deg2rad(theta)
    outer_normals = np.column_stack((np.cos(outer_norm_angles), np.sin(outer_norm_angles)))
    
    max_proj = 0.0
    # All vertices of all inner hexagons
    # _UNIT_HEX_VERTS is (6,2)
    # positions is (n, 2)
    # We can broadcast: verts = positions[:, np.newaxis, :] + _UNIT_HEX_VERTS[np.newaxis, :, :]
    # verts shape: (n, 6, 2)
    verts = positions[:, np.newaxis, :] + _UNIT_HEX_VERTS[np.newaxis, :, :]
    # rel = verts - center
    rel = verts - np.array([cx, cy])
    # rel shape: (n, 6, 2)
    # projs = rel @ outer_normals.T
    # outer_normals.T shape: (2, 6)
    # projs shape: (n, 6, 6)
    projs = np.einsum('ijk,kl->ijl', rel, outer_normals.T)
    # max over vertices (axis 1) and normals (axis 2)
    max_p = np.max(projs)
    side = (2.0 / np.sqrt(3.0)) * max_p
    
    # Calculate penalty
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[i] - positions[j]
            dots = d @ _FORBID_NORMALS.T
            vals = dots - _FORBID_CONST
            max_val = np.max(vals)
            if max_val < 0:
                penalty += (-max_val) ** 2
                
    return side + 10.0 * penalty
