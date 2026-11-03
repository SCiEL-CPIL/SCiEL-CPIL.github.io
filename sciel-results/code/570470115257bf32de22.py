    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Vectorized coarse scan over all 60 possible angles
        phi_vals = np.arange(0, 60, 1) * np.pi / 180.0
        # For each phi, compute the 6 normal angles
        # normal_angles[k, m] = phi[k] + pi/6 + m*pi/3
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6)[None, :] * np.pi/3
        nx = np.cos(normal_angles)  # shape (60, 6)
        ny = np.sin(normal_angles)  # shape (60, 6)
        
        # Project all vertices onto each normal direction
        # We want proj[i, k, m] = offsets[i, 0] * nx[k, m] + offsets[i, 1] * ny[k, m]
        # offsets shape: (72, 2)
        # nx shape: (60, 6)
        
        # Reshape offsets to (72, 1, 1) and nx to (1, 60, 6) to broadcast to (72, 60, 6)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        # proj shape: (72, 60, 6)
        
        abs_proj = np.abs(proj)  # (72, 60, 6)
        max_projs = np.max(abs_proj, axis=0)  # (60, 6)
        side_vals = (2.0 / sqrt3) * max_projs  # (60, 6)
        max_sides = np.max(side_vals, axis=1)  # (60,)
        
        best_idx = np.argmin(max_sides)
        best_side = max_sides[best_idx]
        best_phi = float(phi_vals[best_idx])
        
        # Fine tune with minimize_scalar around the best coarse solution
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            
            def eval_phi(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                # offsets shape (72, 2), nx shape (6,)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                # proj shape (72, 6)
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)
            
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]
