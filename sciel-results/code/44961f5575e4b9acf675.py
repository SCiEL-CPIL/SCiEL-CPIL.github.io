<<<<<<< SEARCH
        # 4. Final high-precision minimize_scalar on exact objective (wider window)
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center

            def eval_phi_final(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)

            try:
                lo = max(0.0, best_phi - 5.0)
                hi = min(60.0, best_phi + 5.0)
                res = minimize_scalar(eval_phi_final, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-12, 'maxiter': 500})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
                    best_cx = center[0]
                    best_cy = center[1]
            except Exception:
                pass
=======
        # 4. Final high-precision minimize_scalar on exact objective
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center

            def eval_phi_final(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)

            try:
                # First: wide scan
                lo_w = max(0.0, best_phi - 8.0)
                hi_w = min(60.0, best_phi + 8.0)
                res_w = minimize_scalar(eval_phi_final, bounds=(lo_w, hi_w), method='bounded',
                                        options={'xatol': 1e-8, 'maxiter': 300})
                if res_w.success:
                    best_phi = res_w.x
                    best_side = res_w.fun

                # Second: tight local refinement
                lo_t = max(0.0, best_phi - 1.0)
                hi_t = min(60.0, best_phi + 1.0)
                res_t = minimize_scalar(eval_phi_final, bounds=(lo_t, hi_t), method='bounded',
                                        options={'xatol': 1e-13, 'maxiter': 1000})
                if res_t.success and res_t.fun < best_side:
                    best_side = res_t.fun
                    best_phi = res_t.x

                best_cx = center[0]
                best_cy = center[1]
            except Exception:
                pass
>>>>>>> REPLACE
