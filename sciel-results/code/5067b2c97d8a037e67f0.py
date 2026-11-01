<<<<<<< SEARCH
    # 5. Final Polish with more diverse starting points and tighter tolerance
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    # Generate diverse polish starting angles
    base_angle = best_outer_angle
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], base_angle],
        [best_outer_center[0], best_outer_center[1], base_angle + 1.0],
        [best_outer_center[0], best_outer_center[1], base_angle - 1.0],
        [best_outer_center[0], best_outer_center[1], base_angle + 5.0],
        [best_outer_center[0], best_outer_center[1], base_angle - 5.0],
        [best_outer_center[0] + 0.05, best_outer_center[1], base_angle],
        [best_outer_center[0] - 0.05, best_outer_center[1], base_angle],
        [best_outer_center[0], best_outer_center[1] + 0.05, base_angle],
        [best_outer_center[0], best_outer_center[1] - 0.05, base_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 20000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (float(best_polish_params[0]), float(best_polish_params[1]))
        best_outer_angle = float(best_polish_params[2])
        best_side = best_polish_val
=======
    # 5. Final Polish with more diverse starting points and tighter tolerance
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    # Generate diverse polish starting angles
    base_angle = best_outer_angle
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], base_angle],
        [best_outer_center[0], best_outer_center[1], base_angle + 1.0],
        [best_outer_center[0], best_outer_center[1], base_angle - 1.0],
        [best_outer_center[0], best_outer_center[1], base_angle + 5.0],
        [best_outer_center[0], best_outer_center[1], base_angle - 5.0],
        [best_outer_center[0] + 0.05, best_outer_center[1], base_angle],
        [best_outer_center[0] - 0.05, best_outer_center[1], base_angle],
        [best_outer_center[0], best_outer_center[1] + 0.05, base_angle],
        [best_outer_center[0], best_outer_center[1] - 0.05, base_angle],
        [best_outer_center[0] + 0.02, best_outer_center[1] + 0.02, base_angle],
        [best_outer_center[0] - 0.02, best_outer_center[1] - 0.02, base_angle],
        [best_outer_center[0] + 0.02, best_outer_center[1] - 0.02, base_angle],
        [best_outer_center[0] - 0.02, best_outer_center[1] + 0.02, base_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 30000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (float(best_polish_params[0]), float(best_polish_params[1]))
        best_outer_angle = float(best_polish_params[2])
        best_side = best_polish_val
    
    # 6. Final high-precision joint polish: fine-tune inner positions AND outer together
    def objective_final_joint(params):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side(inner_list, (ocx, ocy), oangle)
        # Hard check for disjointness
        if not check_all_disjoint(inner_list, eps=1e-6):
            return side + 100.0
        return side
    
    final_params = []
    for x, y, a in best_inner:
        final_params.extend([x, y, a])
    final_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    if time.time() < deadline - 10:
        try:
            res = minimize(objective_final_joint, np.array(final_params), method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 50000})
            if res.fun < best_side and check_all_disjoint(
                [[float(res.x[i*3]), float(res.x[i*3+1]), float(res.x[i*3+2])] for i in range(N)], eps=1e-6):
                best_inner = [[float(res.x[i*3]), float(res.x[i*3+1]), float(res.x[i*3+2])] for i in range(N)]
                best_outer_center = (float(res.x[N*3]), float(res.x[N*3+1]))
                best_outer_angle = float(res.x[N*3+2])
                best_side = float(res.fun)
        except Exception:
            pass
>>>>>>> REPLACE

