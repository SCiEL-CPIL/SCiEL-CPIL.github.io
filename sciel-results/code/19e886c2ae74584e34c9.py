<<<<<<< SEARCH
def _optimize_outer(cx_arr, cy_arr, angle_arr, center, angle_deg, n_center_steps=5, n_angle_steps=6):
    """Grid-search over outer hexagon center and rotation to minimize side length."""
    best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg)
    best_cx, best_cy = center
    best_angle = angle_deg
    
    # Compute centroid of all inner hexagon vertices for initial center guess
    all_verts = []
    for i in range(len(cx_arr)):
        all_verts.extend(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    centroid_x = sum(v[0] for v in all_verts) / len(all_verts)
    centroid_y = sum(v[1] for v in all_verts) / len(all_verts)
    
    # Grid search over center offsets around centroid
    cx0, cy0 = center
    dx_range = 1.5
    dy_range = 1.5
    
    for di in range(n_center_steps):
        for dj in range(n_center_steps):
            tcx = centroid_x + (dx_range * (di - n_center_steps // 2) / max(n_center_steps - 1, 1))
            tcy = centroid_y + (dy_range * (dj - n_center_steps // 2) / max(n_center_steps - 1, 1))
            for angle_idx in range(n_angle_steps):
                tangle = 360.0 * angle_idx / n_angle_steps
                s = _outer_hex_side(cx_arr, cy_arr, angle_arr, (tcx, tcy), tangle)
                if s < best_side:
                    best_side = s
                    best_cx, best_cy = tcx, tcy
                    best_angle = tangle
    
    # Also try original center with different angles
    for angle_idx in range(n_angle_steps):
        tangle = 360.0 * angle_idx / n_angle_steps
        s = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx0, cy0), tangle)
        if s < best_side:
            best_side = s
            best_cx, best_cy = cx0, cy0
            best_angle = tangle
    
    return best_side, (best_cx, best_cy), best_angle
=======
def _optimize_outer(cx_arr, cy_arr, angle_arr, center, angle_deg, n_center_steps=7, n_angle_steps=12):
    """Grid-search over outer hexagon center and rotation to minimize side length."""
    best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg)
    best_cx, best_cy = center
    best_angle = angle_deg
    
    # Compute centroid of all inner hexagon vertices for initial center guess
    all_verts = np.vstack([_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]) for i in range(len(cx_arr))])
    centroid_x = all_verts[:, 0].mean()
    centroid_y = all_verts[:, 1].mean()
    
    # Grid search over center offsets around centroid
    dx_range = 2.0
    dy_range = 2.0
    
    for di in range(n_center_steps):
        for dj in range(n_center_steps):
            tcx = centroid_x + (dx_range * (di - n_center_steps // 2) / max(n_center_steps - 1, 1))
            tcy = centroid_y + (dy_range * (dj - n_center_steps // 2) / max(n_center_steps - 1, 1))
            for angle_idx in range(n_angle_steps):
                tangle = 360.0 * angle_idx / n_angle_steps
                s = _outer_hex_side(cx_arr, cy_arr, angle_arr, (tcx, tcy), tangle)
                if s < best_side:
                    best_side = s
                    best_cx, best_cy = tcx, tcy
                    best_angle = tangle
    
    # Fine-tune with local search around best found
    step = 0.1
    for _ in range(3):
        improved = True
        while improved:
            improved = False
            for dcx in [-step, 0, step]:
                for dcy in [-step, 0, step]:
                    for da in [-5, 0, 5]:
                        tcx = best_cx + dcx
                        tcy = best_cy + dcy
                        tangle = (best_angle + da) % 360
                        s = _outer_hex_side(cx_arr, cy_arr, angle_arr, (tcx, tcy), tangle)
                        if s < best_side - 1e-10:
                            best_side = s
                            best_cx, best_cy = tcx, tcy
                            best_angle = tangle
                            improved = True
        step *= 0.5
    
    return best_side, (best_cx, best_cy), best_angle
>>>>>>> REPLACE

