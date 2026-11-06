    def _soft_covering_radius_vectorized(pts, beta=20.0):
        try:
            hull = ConvexHull(pts)
        except Exception:
            return 180.0
        
        simplices = hull.simplices
        if len(simplices) == 0:
            return 180.0
            
        # Get vertices for all simplices
        # simplices shape (F, 3) usually for 3D hull
        verts = pts[simplices] # Shape (F, 3, 3)
        
        v1 = verts[:, 0, :]
        v2 = verts[:, 1, :]
        v3 = verts[:, 2, :]
        
        e1 = v2 - v1
        e2 = v3 - v1
        
        # Compute Gram matrix components
        # A = [[dot(e1,e1), dot(e2,e1)], [dot(e1,e2), dot(e2,e2)]]
        # b = [dot(e1,e1)/2, dot(e2,e2)/2]
        
        d11 = np.sum(e1 * e1, axis=1)
        d12 = np.sum(e1 * e2, axis=1)
        d22 = np.sum(e2 * e2, axis=1)
        
        # Determinant of A
        det = d11 * d22 - d12 * d12
        
        # Avoid division by zero
        mask = np.abs(det) > 1e-12
        
        if not np.any(mask):
            return 180.0
            
        # Solve for alpha, beta
        # alpha = (b1*d22 - b2*d12) / det
        # beta  = (d11*b2 - d12*b1) / det
        
        b1 = d11 / 2.0
        b2 = d22 / 2.0
        
        alpha = np.zeros_like(det)
        beta_val = np.zeros_like(det)
        
        alpha[mask] = (b1[mask] * d22[mask] - b2[mask] * d12[mask]) / det[mask]
        beta_val[mask] = (d11[mask] * b2[mask] - d12[mask] * b1[mask]) / det[mask]
        
        # Circumcenter pc = v1 + alpha*e1 + beta*e2
        # Note: alpha, beta_val are shape (F,), need to broadcast to (F, 3)
        pc = v1 + alpha[:, np.newaxis] * e1 + beta_val[:, np.newaxis] * e2
        
        pc_norm = np.linalg.norm(pc, axis=1)
        valid = pc_norm > 1e-12
        
        if not np.any(valid):
            return 180.0
            
        cos_R = np.dot(v1[valid], pc[valid], axis=1) / pc_norm[valid]
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        if len(R) == 0:
            return 180.0
            
        max_R = np.max(R)
        if max_R == 0:
            return 0.0
            
        # Soft max
        exp_terms = np.exp(beta * (R - max_R))
        soft_max = max_R + (1.0 / beta) * np.log(np.sum(exp_terms))
        
        return soft_max
    