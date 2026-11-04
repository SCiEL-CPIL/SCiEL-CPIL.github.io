def _smooth_overlap_penalty(centers, angles, outer_theta, n_samples=20):
    penalty = 0.0
    boundary_pts = []
    for i in range(N):
        c = centers[i]
        a = angles[i]
        verts = _hex_vertices(a, c, 1.0)
        pts = []
        n_per_edge = max(1, n_samples // 6)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            for t in np.linspace(0, 1, n_per_edge, endpoint=False):
                p = v1 + t * (v2 - v1)
                pts.append(p)
        boundary_pts.append(np.array(pts))
        
    for i in range(N):
        for j in range(i + 1, N):
            pts_i = boundary_pts[i]
            sdf_vals = _sdf_hexagon_vec(pts_i, centers[j], angles[j])
            neg_sdf = np.maximum(0, -sdf_vals)
            penalty += np.sum(neg_sdf ** 2)
            
            pts_j = boundary_pts[j]
            sdf_vals = _sdf_hexagon_vec(pts_j, centers[i], angles[i])
            neg_sdf = np.maximum(0, -sdf_vals)
            penalty += np.sum(neg_sdf ** 2)
            
    return penalty
