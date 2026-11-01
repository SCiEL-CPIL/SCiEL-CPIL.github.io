<<<<<<< SEARCH
    # 5. Final refinement of outer hexagon with optimized inner
    if time.time() < deadline - 10:
        def objective_final_outer(params):
            cx, cy, angle_deg = params
            side = compute_required_side(best_inner, (cx, cy), angle_deg)
            return side
        
        try:
            res_final = minimize(objective_final_outer, np.array([best_outer_center[0], best_outer_center[1], best_outer_angle]), 
                                method='Nelder-Mead', 
                                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000})
            
            if res_final.fun < best_side:
                ox, oy, oa = res_final.x
                side_final = compute_required_side(best_inner, (ox, oy), oa)
                
                if check_validity(best_inner, (ox, oy), side_final, oa % 360):
                    best_outer_center = (ox, oy)
                    best_outer_angle = oa % 360
                    best_side = side_final
        except Exception:
            pass
=======
    # 5. Final refinement of outer hexagon with optimized inner
    # Use multiple restarts around the current best to find a tighter fit
    if time.time() < deadline - 15:
        def objective_final_outer(params):
            cx, cy, angle_deg = params
            side = compute_required_side(best_inner, (cx, cy), angle_deg)
            return side
        
        best_final = best_side
        best_final_params = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
        
        # Try perturbations around the current best
        perturbations = [
            [0, 0, 0],
            [0.1, 0, 0], [-0.1, 0, 0],
            [0, 0.1, 0], [0, -0.1, 0],
            [0, 0, 5], [0, 0, -5],
            [0.05, 0.05, 0], [-0.05, -0.05, 0],
            [0, 0, 10], [0, 0, -10],
        ]
        
        for dx, dy, da in perturbations:
            if time.time() > deadline - 10:
                break
            
            init_params = np.array([best_outer_center[0] + dx, best_outer_center[1] + dy, 
                                   (best_outer_angle + da) % 360])
            
            try:
                res_final = minimize(objective_final_outer, init_params, 
                                    method='Nelder-Mead', 
                                    options={'maxiter': 600, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 1200})
                
                if res_final.fun < best_final:
                    ox, oy, oa = res_final.x
                    side_final = compute_required_side(best_inner, (ox, oy), oa)
                    
                    if check_validity(best_inner, (ox, oy), side_final, oa % 360):
                        best_final = side_final
                        best_final_params = res_final.x
            except Exception:
                pass
        
        if best_final < best_side:
            ox, oy, oa = best_final_params
            best_outer_center = (ox, oy)
            best_outer_angle = oa % 360
            best_side = best_final
>>>>>>> REPLACE

