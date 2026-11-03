    def construct():
        start_time = time.time()
        time_limit = 115.0  # Safety margin
        
        # 1. Initialization
        # Generate a hexagonal lattice
        s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
        s_gen = s_init * 0.9
        y_step = s_gen * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / s_gen) + 2
        
        points_list = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else s_gen / 2
            for i in range(-max_i, max_i + 1):
                x = i * s_gen + offset
                if np.hypot(x, y) < 0.99:
                    points_list.append([x, y])
                    
        points = np.array(points_list)
        
        if len(points) > N:
            dists_to_center = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists_to_center)[:N]
            points = points[idx]
        elif len(points) < N:
            while len(points) < N:
                p = np.random.uniform(-0.9, 0.9, 2)
                points = np.vstack([points, p])
            points = points[:N]
            
        best_points = points.copy()
        best_score = compute_score(points)
        
        # 2. Simulated Annealing Phase
        # We use a multi-start approach with SA.
        # Given time constraints, we will run a few restarts.
        # Each restart: Perturb current best, run SA.
        
        num_restarts = 5
        time_per_restart = (time_limit - (time.time() - start_time)) / num_restarts
        
        for r in range(num_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Initialize for this restart
            if r == 0:
                current_points = best_points.copy()
            else:
                # Perturb best points to escape local minima
                current_points = best_points + np.random.normal(0, 0.05, size=best_points.shape)
                # Project inside
                rad = np.linalg.norm(current_points, axis=1)
                mask = rad > 1.0
                if np.any(mask):
                    current_points[mask] /= rad[mask, np.newaxis]
                    
            current_score = compute_score(current_points)
            
            # SA Parameters
            T_start = 0.05
            T_end = 0.001
            alpha = 0.995
            T = T_start
            max_sa_iters = 2000 # Adjust based on time
            
            # Estimate time per SA iter
            # compute_score is O(N^2). N=300 -> 90k ops. Very fast.
            # We can run many iters.
            
            sa_start_time = time.time()
            sa_time_limit = time_per_restart * 0.7 # Leave time for polish
            
            for it in range(max_sa_iters):
                if time.time() - sa_start_time > sa_time_limit:
                    break
                    
                # Pick a random point to move
                idx = np.random.randint(N)
                old_point = current_points[idx].copy()
                
                # Random move
                move_size = T * 0.5
                new_point = old_point + np.random.normal(0, move_size, 2)
                
                # Check if inside container
                if np.linalg.norm(new_point) >= 1.0:
                    # Reject or project? Reject is simpler for SA
                    continue
                    
                current_points[idx] = new_point
                new_score = compute_score(current_points)
                
                # Acceptance
                if new_score > current_score:
                    current_score = new_score
                else:
                    delta = new_score - current_score
                    if np.random.rand() < np.exp(delta / T):
                        current_score = new_score
                    else:
                        current_points[idx] = old_point
                        
                T *= alpha
                
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
            # 3. Polish Phase for this candidate
            # Use remaining time for this restart
            polish_time = time_per_restart * 0.3
            polished_points, polished_score = polish(current_points, polish_time)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points
                
            # Try least_squares polish if time remains
            remaining_time_ls = time_per_restart * 0.3 - (time.time() - start_time - (time_limit - (time_limit - polish_time))) # This logic is messy
            # Let's just check if we have time before the next restart
            if time.time() - start_time < time_limit - 5.0:
                ls_time = min(2.0, time_limit - 5.0 - (time.time() - start_time))
                if ls_time > 0.5:
                    ls_points, ls_score = polish_with_least_squares(best_points, ls_time)
                    if ls_score > best_score:
                        best_score = ls_score
                        best_points = ls_points
                        
        # Final global polish on the best found
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 1.0:
            final_points, final_score = polish(best_points, remaining_time)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
                
        # Additional high-precision polish if time remains
        remaining_time2 = time_limit - (time.time() - start_time)
        if remaining_time2 > 2.0:
            # Try least_squares polish first
            ls_time = min(5.0, remaining_time2 - 1.0)
            if ls_time > 0.5:
                ls_points, ls_score = polish_with_least_squares(best_points, ls_time)
                if ls_score > best_score:
                    best_score = ls_score
                    best_points = ls_points
                    
            # Then very short final polish with even sharper soft-min
            remaining_time3 = time_limit - (time.time() - start_time)
            if remaining_time3 > 1.0:
                i_idx2, j_idx2 = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
                num_pairs2 = len(i_idx2)
                p = best_points.copy()
                k2 = 1000.0
                lr2 = 1e-5
                polish_start = time.time()
                
                while time.time() - polish_start < remaining_time3 - 1.0:
                    diff2 = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                    dists2 = np.sqrt(np.sum(diff2**2, axis=2))
                    np.fill_diagonal(dists2, np.inf)
                    
                    pair_d2 = dists2[i_idx2, j_idx2]
                    rad2 = np.linalg.norm(p, axis=1)
                    bound2 = 1.0 - rad2
                    
                    all_d2 = np.concatenate([pair_d2, bound2])
                    cur_score = np.min(all_d2)
                    
                    shifted2 = all_d2 - cur_score
                    mask2 = shifted2 < 0.001
                    if not np.any(mask2):
                        break
                        
                    close2 = shifted2[mask2]
                    exp_w2 = np.exp(-k2 * close2)
                    sum_w2 = np.sum(exp_w2)
                    if sum_w2 == 0:
                        break
                        
                    weights2 = np.zeros_like(all_d2)
                    weights2[mask2] = exp_w2 / sum_w2
                    
                    grads2 = np.zeros_like(p)
                    
                    pw2 = weights2[:num_pairs2]
                    d_p2 = dists2[i_idx2, j_idx2]
                    mn2 = d_p2 > 1e-9
                    if np.any(mn2):
                        v2 = diff2[i_idx2[mn2], j_idx2[mn2]] / d_p2[mn2, np.newaxis]
                        w2 = pw2[mn2, np.newaxis]
                        f2 = np.zeros((N, 2))
                        np.add.at(f2, i_idx2[mn2], v2 * w2)
                        np.add.at(f2, j_idx2[mn2], -v2 * w2)
                        grads2 += f2
                        
                    bw2 = weights2[num_pairs2:]
                    rs2 = np.where(rad2 > 1e-9, rad2, 1.0)
                    grads2 += -(p / rs2[:, np.newaxis]) * bw2[:, np.newaxis]
                    
                    gn2 = np.linalg.norm(grads2)
                    if gn2 > 1e-9:
                        grads2 /= gn2
                        p += lr2 * grads2
                        
                    rn2 = np.linalg.norm(p, axis=1)
                    m2 = rn2 > 1.0
                    if np.any(m2):
                        p[m2] /= rn2[m2, np.newaxis]
                        
                    lr2 *= 0.999
                    k2 *= 1.001
                    
                # Validate and update
                rad_final = np.linalg.norm(p, axis=1)
                if np.all(rad_final <= 1.0) and np.all(np.isfinite(p)):
                    diff_f = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                    dists_f = np.sqrt(np.sum(diff_f**2, axis=2))
                    np.fill_diagonal(dists_f, np.inf)
                    min_pair_f = np.min(dists_f)
                    min_bound_f = 1.0 - np.max(rad_final)
                    score_f = min(min_pair_f, min_bound_f)
                    
                    if score_f > best_score:
                        best_score = score_f
                        best_points = p
                
        return best_points
    