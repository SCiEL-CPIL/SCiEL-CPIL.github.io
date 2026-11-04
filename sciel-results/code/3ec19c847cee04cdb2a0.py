        # Precompute vertices
        vertices = []
        for i in range(N):
            x, y, th = inner_params[i]
            v = get_hex_vertices((x, y), th)
            vertices.append(v)
            
        for i in range(N):
            for j in range(i + 1, N):
                d = poly_dist(vertices[i], vertices[j])
                # If d > 0, disjoint.
                # If d == 0, they touch or overlap.
                # We need a penalty for overlap.
                # If they overlap, d is 0.
                # We can't distinguish touch from overlap with just d.
                # But if they touch, d=0 is acceptable.
                # If they overlap, d=0 is bad.
                # To penalize overlap, we can check if they intersect.
                # If they intersect, we can use a small penalty, or estimate depth.
                # Let's assume if d < 1e-6, they are touching/overlapping.
                # To encourage separation, we can use a penalty if d < epsilon.
                # But Nelder-Mead needs a gradient.
                # A better penalty: if they overlap, use the center distance heuristic as a proxy for depth.
                
                # Let's combine:
                # If d > 0: penalty 0.
                # If d <= 0: 
                #   Check if they actually overlap (intersect).
                #   If yes, penalty = (sqrt(3) - d_c)^2.
                #   If no (touching), penalty 0.
                
                # Checking intersection is expensive.
                # Let's just use the center distance heuristic for ALL pairs.
                # It's simpler and likely sufficient for Nelder-Mead to find a good local min.
                # The exact geometry is handled by the containment check and the fact that we minimize L.
                
                cx1, cy1, _ = inner_params[i]
                cx2, cy2, _ = inner_params[j]
                d_c = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                overlap = np.sqrt(3) - d_c
                if overlap > 0:
                    penalty += overlap * overlap
