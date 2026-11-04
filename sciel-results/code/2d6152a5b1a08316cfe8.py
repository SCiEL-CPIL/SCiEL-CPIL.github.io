<<<<<<< SEARCH
    # Coordinate descent polish
    remaining = DEADLINE - (time.time() - t0)
    
    if remaining > 5.0:
        cd_deadline = t0 + remaining * 0.5
        step = 0.03
        min_step = 0.0001
        
        while time.time() < cd_deadline and step > min_step:
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()
                        
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
                        
                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.6
            else:
                step = max(step * 0.8, min_step)
        
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
=======
    # Coordinate descent polish with discrete rotation
    remaining = DEADLINE - (time.time() - t0)
    
    if remaining > 5.0:
        cd_deadline = t0 + remaining * 0.5
        step = 0.03
        min_step = 0.0001
        
        while time.time() < cd_deadline and step > min_step:
            improved = False
            for idx in range(N):
                # Try position moves
                for dim in range(2):
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()
                        
                        if dim == 0:
                            new_pos[idx, 0] += direction * step
                        else:
                            new_pos[idx, 1] += direction * step
                        
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
                        
                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                
                # Try discrete rotation changes
                if not improved:
                    current_rot_idx = np.argmin(np.abs(ROT_ANGLES_DEG - (best_angles[idx] % 360)))
                    for new_rot_idx in range(N_ROT_STATES):
                        if new_rot_idx == current_rot_idx:
                            continue
                        
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()
                        new_ang[idx] = ROT_ANGLES_DEG[new_rot_idx]
                        
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
                        
                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            break
                
                if improved:
                    break
            if not improved:
                step *= 0.6
            else:
                step = max(step * 0.8, min_step)
        
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
>>>>>>> REPLACE

