<<<<<<< SEARCH
    # Phase 3: High-precision polish
    if time.time() < deadline - 20:
        current_inner = best_result[0].copy()
        current_s = best_result[2]
        
        polish_params = np.zeros(25)
        polish_params[:24] = current_inner[:, :2].flatten()
        polish_params[24] = current_s
        
        try:
            polish_result = minimize(
                objective,
                polish_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if polish_result.fun < 100:
                final_params = polish_result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_xy_final, inner_angles)
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception:
            pass
    
    return best_result
=======
    # Phase 3: High-precision polish
    if time.time() < deadline - 20:
        current_inner = best_result[0].copy()
        current_s = best_result[2]
        
        polish_params = np.zeros(25)
        polish_params[:24] = current_inner[:, :2].flatten()
        polish_params[24] = current_s
        
        try:
            polish_result = minimize(
                objective,
                polish_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if polish_result.fun < 100:
                final_params = polish_result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_xy_final, inner_angles)
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception:
            pass
    
    # Phase 4: Final tight polish if time remains
    if time.time() < deadline - 5:
        current_inner = best_result[0].copy()
        current_s = best_result[2]
        
        polish_params = np.zeros(25)
        polish_params[:24] = current_inner[:, :2].flatten()
        polish_params[24] = current_s
        
        try:
            polish_result = minimize(
                objective,
                polish_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if polish_result.fun < 100:
                final_params = polish_result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_xy_final, inner_angles)
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception:
            pass
    
    return best_result
>>>>>>> REPLACE
