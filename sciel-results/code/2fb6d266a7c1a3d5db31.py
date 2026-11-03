<<<<<<< SEARCH
    # Run SA from the best initial configuration
    sa_time_budget = DEADLINE * 0.70
=======
    # Run SA from the best initial configuration
    sa_time_budget = DEADLINE * 0.65
    
    # Add gradient-based refinement step after spectral initialization
    def gradient_refine(pos, ang, time_limit):
        """Refine positions using gradient descent on the outer side length."""
        current_side, _, _, _ = compute_min_outer_side(pos, ang)
        best_side = current_side
        best_pos = pos.copy()
        best_ang = ang.copy()
        
        t_start = time.time()
        lr = 0.02
        min_lr = 0.0001
        no_improve = 0
        max_no_improve = 50
        
        while time.time() - t_start < time_limit and lr > min_lr:
            improved = False
            for idx in range(N):
                if time.time() - t_start > time_limit:
                    break
                # Numerical gradient for x, y, angle
                eps = 0.005
                grad = np.zeros(3)
                for dim in range(3):
                    pos_plus = pos.copy()
                    ang_plus = ang.copy()
                    if dim == 0:
                        pos_plus[idx, 0] += eps
                    elif dim == 1:
                        pos_plus[idx, 1] += eps
                    else:
                        ang_plus[idx] += eps
                    angles_rad_plus = np.radians(ang_plus)
                    valid_plus = True
                    h_moved_plus = (pos_plus[idx, 0], pos_plus[idx, 1], angles_rad_plus[idx])
                    for j in range(N):
                        if j == idx: continue
                        h_other = (pos_plus[j, 0], pos_plus[j, 1], angles_rad_plus[j])
                        if not hexagons_disjoint_fast(h_moved_plus, h_other):
                            valid_plus = False
                            break
                    if valid_plus:
                        side_plus, _, _, _ = compute_min_outer_side(pos_plus, ang_plus)
                    else:
                        side_plus = best_side + 1.0
                    
                    pos_minus = pos.copy()
                    ang_minus = ang.copy()
                    if dim == 0:
                        pos_minus[idx, 0] -= eps
                    elif dim == 1:
                        pos_minus[idx, 1] -= eps
                    else:
                        ang_minus[idx] -= eps
                    angles_rad_minus = np.radians(ang_minus)
                    valid_minus = True
                    h_moved_minus = (pos_minus[idx, 0], pos_minus[idx, 1], angles_rad_minus[idx])
                    for j in range(N):
                        if j == idx: continue
                        h_other = (pos_minus[j, 0], pos_minus[j, 1], angles_rad_minus[j])
                        if not hexagons_disjoint_fast(h_moved_minus, h_other):
                            valid_minus = False
                            break
                    if valid_minus:
                        side_minus, _, _, _ = compute_min_outer_side(pos_minus, ang_minus)
                    else:
                        side_minus = best_side + 1.0
                    
                    grad[dim] = (side_plus - side_minus) / (2 * eps)
                
                # Take gradient step
                new_pos = pos.copy()
                new_ang = ang.copy()
                new_pos[idx, 0] -= lr * grad[0]
                new_pos[idx, 1] -= lr * grad[1]
                new_ang[idx] -= lr * grad[2]
                
                angles_rad_new = np.radians(new_ang)
                valid_new = True
                h_moved_new = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                for j in range(N):
                    if j == idx: continue
                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                    if not hexagons_disjoint_fast(h_moved_new, h_other):
                        valid_new = False
                        break
                
                if valid_new:
                    new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side - 1e-12:
                        pos = new_pos
                        ang = new_ang
                        best_side = new_side
                        best_pos = new_pos.copy()
                        best_ang = new_ang.copy()
                        improved = True
                        no_improve = 0
                        break
                else:
                    no_improve += 1
            
            if not improved:
                no_improve += 1
                lr *= 0.8
                if no_improve > max_no_improve:
                    break
            else:
                lr *= 0.95
                no_improve = 0
        
        return best_side, best_pos, best_ang
    
    # Apply gradient refinement before SA
    grad_time = min(remaining_time * 0.15, 8.0)
    grad_side, grad_pos, grad_ang = gradient_refine(best_positions, best_angles, grad_time)
    if grad_side < best_side:
        best_side = grad_side
        best_positions = grad_pos
        best_angles = grad_ang
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
>>>>>>> REPLACE

