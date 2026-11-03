<<<<<<< SEARCH
    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
=======
    # Gradient-based refinement using support function gradients
    def gradient_refine(pos, ang, time_limit):
        """
        Refine positions using gradient descent on the outer hexagon side length.
        Moves hexagons away from critical vertices to reduce the support function.
        """
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(pos, ang)
        best_side = current_side
        best_pos = pos.copy()
        best_ang = ang.copy()
        
        t_start = time.time()
        lr = 0.05
        min_lr = 0.001
        no_improve = 0
        max_no_improve = 30
        
        while time.time() - t_start < time_limit and lr > min_lr:
            improved = False
            
            # Compute the critical direction
            angles_rad = np.radians(ang)
            all_verts = get_all_vertices(pos, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            ox = offsets[:, 0]
            oy = offsets[:, 1]
            
            normal_offsets = np.pi/6 + np.arange(6) * np.pi/3
            phi_rad = np.radians(current_phi)
            normal_angles = phi_rad + normal_offsets
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            proj = ox[:, None] * nx[None, :] + oy[:, None] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            critical_norm_idx = np.argmax(side_vals)
            
            # For each hexagon, determine gradient direction
            gradients = np.zeros((N, 2))
            for i in range(N):
                vert_start = i * 6
                vert_end = (i+1) * 6
                vert_ox = ox[vert_start:vert_end]
                vert_oy = oy[vert_start:vert_end]
                
                proj_i = vert_ox * nx[critical_norm_idx] + vert_oy * ny[critical_norm_idx]
                abs_proj_i = np.abs(proj_i)
                max_vert_idx = np.argmax(abs_proj_i)
                
                sign = 1.0 if proj_i[max_vert_idx] > 0 else -1.0
                
                # Move opposite to the critical normal to reduce support
                gradients[i, 0] = -sign * nx[critical_norm_idx]
                gradients[i, 1] = -sign * ny[critical_norm_idx]
            
            grad_norms = np.linalg.norm(gradients, axis=1, keepdims=True)
            grad_norms[grad_norms < 1e-10] = 1.0
            gradients = gradients / grad_norms
            
            new_pos = pos.copy()
            for i in range(N):
                new_pos[i, 0] += lr * gradients[i, 0]
                new_pos[i, 1] += lr * gradients[i, 1]
            
            angles_rad_new = np.radians(ang)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad_new[i])
                    h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                    if not hexagons_disjoint_fast(h1, h2):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, ang)
                if new_side < best_side - 1e-10:
                    best_side = new_side
                    best_pos = new_pos
                    best_ang = ang.copy()
                    improved = True
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            if improved:
                lr *= 0.95
            else:
                lr *= 0.8
                if no_improve > max_no_improve:
                    break
        
        return best_side, best_pos, best_ang
    
    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
>>>>>>> REPLACE

