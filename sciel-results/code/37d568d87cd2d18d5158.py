def check_overlap_penalty(centers, angles):
    penalty = 0.0
    verts = np.zeros((N, 6, 2))
    norms = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i] = get_hex_vertices(centers[i], angles[i])
        norms[i] = get_hex_normals(angles[i])
        
    for i in range(N):
        for j in range(i + 1, N):
            dist_sq = np.sum((centers[i] - centers[j])**2)
            if dist_sq > 4.0:
                continue
                
            min_sep = float('inf')
            
            # Check normals of i
            p1 = verts[i] @ norms[i].T
            p2 = verts[j] @ norms[i].T
            seps = np.maximum(p1.min(axis=0) - p2.max(axis=0), p2.min(axis=0) - p1.max(axis=0))
            min_sep = np.min(seps)
            if min_sep > 1e-8:
                continue
                
            # Check normals of j
            p1 = verts[i] @ norms[j].T
            p2 = verts[j] @ norms[j].T
            seps = np.maximum(p1.min(axis=0) - p2.max(axis=0), p2.min(axis=0) - p1.max(axis=0))
            min_sep = np.min(seps)
            
            if min_sep < 0:
                penalty += (-min_sep)**2
    return penalty
