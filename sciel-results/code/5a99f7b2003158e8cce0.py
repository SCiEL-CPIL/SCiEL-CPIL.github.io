    def construct():
        start_time = time.time()
        
        axial_coords = [
            (0, 0),
            (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
            (2, 0), (2, -1), (2, -2), (1, -2)
        ]
        
        centers_rel = []
        for q, r in axial_coords:
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            centers_rel.append((x, y))
        
        base_angles = np.arange(6) * np.pi / 3
        base_vx = np.cos(base_angles)
        base_vy = np.sin(base_angles)
        
        V_rel = np.zeros((11, 6, 2))
        for i, (cx, cy) in enumerate(centers_rel):
            V_rel[i, :, 0] = cx + base_vx
            V_rel[i, :, 1] = cy + base_vy
        
        verts_flat = V_rel.reshape(66, 2)
        
        theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(theta_n)
        ny = np.sin(theta_n)
        apothem = np.sqrt(3) / 2
        
        def fast_objective(params):
            cx, cy, alpha = params
            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)
            
            rx = verts_flat[:, 0] * cos_a - verts_flat[:, 1] * sin_a + cx
            ry = verts_flat[:, 0] * sin_a + verts_flat[:, 1] * cos_a + cy
            
            proj = rx[np.newaxis, :] * nx[:, np.newaxis] + ry[np.newaxis, :] * ny[:, np.newaxis]
            max_proj = np.max(np.abs(proj))
            
            return max_proj / apothem
        
        best_params = None
        best_S = float('inf')
        
        # Try a few initial rotations
        for init_alpha in [0.0, np.pi/6, np.pi/3, np.pi/2]:
            p0 = [0.0, 0.0, init_alpha]
            res = minimize(fast_objective, p0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        
        cx, cy, alpha = best_params
        
        inner = []
        for i, (rel_x, rel_y) in enumerate(centers_rel):
            g_x = rel_x * np.cos(alpha) - rel_y * np.sin(alpha) + cx
            g_y = rel_x * np.sin(alpha) + rel_y * np.cos(alpha) + cy
            inner.append([g_x, g_y, np.degrees(alpha)])
            
        return inner, (0.0, 0.0), best_S, 0.0
    