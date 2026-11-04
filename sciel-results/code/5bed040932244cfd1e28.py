<<<<<<< SEARCH
            if not valid:
                T *= cooling
                if T < T_min: T = T_min
                step_size *= step_decay
                if step_size < 0.005: step_size = 0.005
                continue
                
            # Use fast evaluation during SA (no fine tune)
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles, fine_tune=False)
            
            if new_side < current_side:
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                if new_side < best_side:
                    # Recompute with fine tune for best tracking
                    fs, fp, fcx, fcy = compute_min_outer_side(new_positions, new_angles, fine_tune=True)
                    if fs < best_side:
                        best_side = fs
                        best_positions = new_positions.copy()
                        best_angles = new_angles.copy()
                        best_phi = fp
                        best_cx = fcx
                        best_cy = fcy
            else:
                delta = new_side - current_side
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / T):
                    current_positions = new_positions
                    current_angles = new_angles
                    current_side = new_side
                    current_phi = new_phi
                    current_cx = new_cx
                    current_cy = new_cy
                    accept_count += 1
                    if new_side < best_side:
                        fs, fp, fcx, fcy = compute_min_outer_side(new_positions, new_angles, fine_tune=True)
                        if fs < best_side:
                            best_side = fs
                            best_positions = new_positions.copy()
                            best_angles = new_angles.copy()
                            best_phi = fp
                            best_cx = fcx
                            best_cy = fcy
            
            # Adaptive step size
            if total_count % 500 == 0:
                acc_rate = accept_count / total_count
                if acc_rate < 0.1:
                    step_size *= 0.8
                    if step_size < 0.005: step_size = 0.005
                elif acc_rate > 0.5:
                    step_size *= 1.2
                    if step_size > 1.0: step_size = 1.0
                accept_count = 0
                total_count = 0
                
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005
=======
            if not valid:
                T *= cooling
                if T < T_min: T = T_min
                step_size *= step_decay
                if step_size < 0.005: step_size = 0.005
                continue
                
            # Use fast evaluation during SA (no fine tune)
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles, fine_tune=False)
            
            if new_side < current_side:
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                last_improve_step = step_count
                if new_side < best_side:
                    # Recompute with fine tune for best tracking
                    fs, fp, fcx, fcy = compute_min_outer_side(new_positions, new_angles, fine_tune=True)
                    if fs < best_side:
                        best_side = fs
                        best_positions = new_positions.copy()
                        best_angles = new_angles.copy()
                        best_phi = fp
                        best_cx = fcx
                        best_cy = fcy
            else:
                delta = new_side - current_side
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / T):
                    current_positions = new_positions
                    current_angles = new_angles
                    current_side = new_side
                    current_phi = new_phi
                    current_cx = new_cx
                    current_cy = new_cy
                    accept_count += 1
                    if new_side < best_side:
                        fs, fp, fcx, fcy = compute_min_outer_side(new_positions, new_angles, fine_tune=True)
                        if fs < best_side:
                            best_side = fs
                            best_positions = new_positions.copy()
                            best_angles = new_angles.copy()
                            best_phi = fp
                            best_cx = fcx
                            best_cy = fcy
            
            # Adaptive step size
            if total_count % 300 == 0:
                acc_rate = accept_count / total_count
                if acc_rate < 0.05:
                    step_size *= 0.7
                    if step_size < 0.003: step_size = 0.003
                elif acc_rate > 0.4:
                    step_size *= 1.3
                    if step_size > 1.0: step_size = 1.0
                accept_count = 0
                total_count = 0
                
            # Reheat if stuck
            if step_count - last_improve_step > 2000 and reheat_count < max_reheats:
                T = max(T, 0.3)
                step_size = max(step_size, 0.15)
                reheat_count += 1
                last_improve_step = step_count
            
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003
>>>>>>> REPLACE

