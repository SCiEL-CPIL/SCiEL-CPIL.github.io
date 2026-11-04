<<<<<<< SEARCH
    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 3.0:
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang, fine_tune=False)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.05
            min_step = 0.0005
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_pos_cd.copy()
                            new_ang = best_ang_cd.copy()
                            
                            if dim == 0:
                                new_pos[idx, 0] += direction * step
                            elif dim == 1:
                                new_pos[idx, 1] += direction * step
                            else:
                                new_ang[idx] += direction * step
                            
                            angles_rad_new = np.radians(new_ang)
                            valid = True
                            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                            for j in range(N):
                                if j == idx: continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)
                            
                            if new_side < best_side_cd - 1e-12:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                break
                        
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    step *= 0.7
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.4, 15.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 10.0:
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                res = minimize(smooth_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
                                        'adaptive': True})
                
                if res.fun < 1e5:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
                            init_params = res.x.copy()
                
                remaining_time2 = DEADLINE - (time.time() - t0)
                if remaining_time2 > 5.0:
                    def exact_objective(params):
                        pos = np.zeros((N, 2))
                        ang = np.zeros(N)
                        for i in range(N):
                            pos[i, 0] = params[3*i]
                            pos[i, 1] = params[3*i+1]
                            ang[i] = params[3*i+2]
                        
                        angles_rad = np.radians(ang)
                        for i in range(N):
                            for j in range(i+1, N):
                                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                                if not hexagons_disjoint_fast(h1, h2):
                                    return 1e6
                        
                        side, phi, cx, cy = compute_min_outer_side(pos, ang, fine_tune=True)
                        return side
                    
                    res2 = minimize(exact_objective, init_params, method='Nelder-Mead', 
                                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                             'adaptive': True})
                    
                    if res2.fun < 1e5:
                        new_pos = np.zeros((N, 2))
                        new_ang = np.zeros(N)
                        for i in range(N):
                            new_pos[i, 0] = res2.x[3*i]
                            new_pos[i, 1] = res2.x[3*i+1]
                            new_ang[i] = res2.x[3*i+2]
                        
                        if is_valid_configuration(new_pos, new_ang):
                            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                            if new_side < best_side:
                                best_side = new_side
                                best_positions = new_pos
                                best_angles = new_ang
                                best_phi = new_phi
                                best_cx = new_cx
                                best_cy = new_cy
            except Exception:
                pass
        
        remaining_time3 = DEADLINE - (time.time() - t0)
        if remaining_time3 > 3.0:
            final_step = 0.01
            final_min_step = 0.0001
            t_start_final = time.time()
            
            while time.time() - t_start_final < remaining_time3 - 1.0 and final_step > final_min_step:
                improved_final = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_positions.copy()
                            new_ang = best_angles.copy()
                            
                            if dim == 0:
                                new_pos[idx, 0] += direction * final_step
                            elif dim == 1:
                                new_pos[idx, 1] += direction * final_step
                            else:
                                new_ang[idx] += direction * final_step
                            
                            angles_rad_new = np.radians(new_ang)
                            valid = True
                            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                            for j in range(N):
                                if j == idx: continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)
                            
                            if new_side < best_side - 1e-13:
                                best_positions = new_pos
                                best_angles = new_ang
                                best_side = new_side
                                improved_final = True
                                break
                        
                        if time.time() - t_start_final > remaining_time3 - 1.0:
                            break
                    else:
                        continue
                    break
                else:
                    final_step *= 0.6
                    continue
                break
            
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
=======
    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 3.0:
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang, fine_tune=False)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.03
            min_step = 0.0001
            no_improve_count = 0
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                no_improve_count = 0
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_pos_cd.copy()
                            new_ang = best_ang_cd.copy()
                            
                            if dim == 0:
                                new_pos[idx, 0] += direction * step
                            elif dim == 1:
                                new_pos[idx, 1] += direction * step
                            else:
                                new_ang[idx] += direction * step
                            
                            angles_rad_new = np.radians(new_ang)
                            valid = True
                            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                            for j in range(N):
                                if j == idx: continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)
                            
                            if new_side < best_side_cd - 1e-14:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                break
                            else:
                                no_improve_count += 1
                        
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    if no_improve_count > N * 3 * 2:
                        step *= 0.5
                    else:
                        step *= 0.7
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.35, 12.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 8.0:
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                # First Nelder-Mead with smooth surrogate
                res = minimize(smooth_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                
                if res.fun < 1e5:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
                            init_params = res.x.copy()
                
                remaining_time2 = DEADLINE - (time.time() - t0)
                if remaining_time2 > 5.0:
                    def exact_objective(params):
                        pos = np.zeros((N, 2))
                        ang = np.zeros(N)
                        for i in range(N):
                            pos[i, 0] = params[3*i]
                            pos[i, 1] = params[3*i+1]
                            ang[i] = params[3*i+2]
                        
                        angles_rad = np.radians(ang)
                        for i in range(N):
                            for j in range(i+1, N):
                                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                                if not hexagons_disjoint_fast(h1, h2):
                                    return 1e6
                        
                        side, phi, cx, cy = compute_min_outer_side(pos, ang, fine_tune=True)
                        return side
                    
                    res2 = minimize(exact_objective, init_params, method='Nelder-Mead', 
                                    options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11,
                                             'adaptive': True})
                    
                    if res2.fun < 1e5:
                        new_pos = np.zeros((N, 2))
                        new_ang = np.zeros(N)
                        for i in range(N):
                            new_pos[i, 0] = res2.x[3*i]
                            new_pos[i, 1] = res2.x[3*i+1]
                            new_ang[i] = res2.x[3*i+2]
                        
                        if is_valid_configuration(new_pos, new_ang):
                            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                            if new_side < best_side:
                                best_side = new_side
                                best_positions = new_pos
                                best_angles = new_ang
                                best_phi = new_phi
                                best_cx = new_cx
                                best_cy = new_cy
            except Exception:
                pass
        
        # Final high-precision coordinate descent
        remaining_time3 = DEADLINE - (time.time() - t0)
        if remaining_time3 > 2.0:
            final_step = 0.005
            final_min_step = 0.00005
            t_start_final = time.time()
            
            while time.time() - t_start_final < remaining_time3 - 1.0 and final_step > final_min_step:
                improved_final = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_positions.copy()
                            new_ang = best_angles.copy()
                            
                            if dim == 0:
                                new_pos[idx, 0] += direction * final_step
                            elif dim == 1:
                                new_pos[idx, 1] += direction * final_step
                            else:
                                new_ang[idx] += direction * final_step
                            
                            angles_rad_new = np.radians(new_ang)
                            valid = True
                            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                            for j in range(N):
                                if j == idx: continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=True)
                            
                            if new_side < best_side - 1e-14:
                                best_positions = new_pos
                                best_angles = new_ang
                                best_side = new_side
                                improved_final = True
                                break
                        
                        if time.time() - t_start_final > remaining_time3 - 1.0:
                            break
                    else:
                        continue
                    break
                else:
                    final_step *= 0.5
                    continue
                break
            
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
>>>>>>> REPLACE

