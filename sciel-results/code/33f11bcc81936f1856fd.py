<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    # Strategy:
    # 1. L-BFGS-B with penalty to get close to feasible region.
    # 2. SLSQP with hard constraints to maximize radii within feasibility.
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: L-BFGS-B with penalty (use smooth objective for better gradient info)
        penalty_weights = [50.0, 500.0, 2000.0]
        for p_w in penalty_weights:
            if time.time() > deadline - 18:
                break
            try:
                # Use smooth objective for L-BFGS-B
                result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-7})
                if result.fun < _objective_smooth(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Stage 2: SLSQP with hard constraints
        pos_start = current_x[:2*N].reshape(N, 2)
        radii_start = np.maximum(current_x[2*N:], 1e-4)
        pos_start, radii_start = _post_process(pos_start, radii_start)
        
        if time.time() < deadline - 8:
            try:
                x_slsqp0 = np.concatenate([pos_start.flatten(), radii_start])
                cons = _slsqp_constraints(x_slsqp0)
                
                result = minimize(_objective_slsqp, x_slsqp0, method='SLSQP', bounds=bounds,
                                  constraints=cons,
                                  options={'maxiter': 250, 'ftol': 1e-14})
                
                if result.success or result.fun < _objective_slsqp(x_slsqp0):
                    pos_sl = result.x[:2*N].reshape(N, 2)
                    radii_sl = np.maximum(result.x[2*N:], 1e-4)
                    pos_pp, radii_pp = _post_process(pos_sl, radii_sl)
                    score = np.sum(radii_pp)
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
                        
            except Exception:
                pass
    
    # Final global polish on the best result found
    if time.time() < deadline - 5:
        try:
            pos_final, radii_final = _post_process(best_pos, best_radii)
            x_final0 = np.concatenate([pos_final.flatten(), radii_final])
            cons_final = _slsqp_constraints(x_final0)
            
            # Try to push further with SLSQP
            result = minimize(_objective_slsqp, x_final0, method='SLSQP', bounds=bounds,
                              constraints=cons_final,
                              options={'maxiter': 300, 'ftol': 1e-14})
            
            if result.success:
                pos_pf = result.x[:2*N].reshape(N, 2)
                radii_pf = np.maximum(result.x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_pf, radii_pf)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
            
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to a known valid configuration
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result
=======
def _qp_radius_optimize(pos, radii_init, max_iter=50, tol=1e-10):
    """
    Given fixed positions, find maximum radii via iterative water-filling.
    For each circle, its max radius is limited by walls and neighbors.
    We iteratively solve: r_i = min(wall_i, min_j (dist_ij - r_j))
    Using successive over-relaxation for convergence.
    """
    radii = radii_init.copy()
    
    # Precompute distances
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Wall limits
    wall = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                      np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    
    for it in range(max_iter):
        old_radii = radii.copy()
        
        # For each circle, compute its max possible radius
        for i in range(N):
            # Wall constraint
            max_r = wall[i]
            # Neighbor constraints: dist[i,j] - r_j
            for j in range(N):
                if i != j:
                    max_r = min(max_r, dist[i, j] - radii[j])
            radii[i] = max(max_r, 1e-6)
        
        # Check convergence
        if np.max(np.abs(radii - old_radii)) < tol:
            break
    
    # Ensure strict validity with small shrink
    radii *= 0.9999
    return radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    # Strategy:
    # 1. L-BFGS-B with penalty to get close to feasible region.
    # 2. SLSQP with hard constraints to maximize radii within feasibility.
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: L-BFGS-B with penalty (use smooth objective for better gradient info)
        penalty_weights = [50.0, 500.0, 2000.0]
        for p_w in penalty_weights:
            if time.time() > deadline - 18:
                break
            try:
                # Use smooth objective for L-BFGS-B
                result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-7})
                if result.fun < _objective_smooth(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Stage 2: SLSQP with hard constraints
        pos_start = current_x[:2*N].reshape(N, 2)
        radii_start = np.maximum(current_x[2*N:], 1e-4)
        pos_start, radii_start = _post_process(pos_start, radii_start)
        
        if time.time() < deadline - 8:
            try:
                x_slsqp0 = np.concatenate([pos_start.flatten(), radii_start])
                cons = _slsqp_constraints(x_slsqp0)
                
                result = minimize(_objective_slsqp, x_slsqp0, method='SLSQP', bounds=bounds,
                                  constraints=cons,
                                  options={'maxiter': 250, 'ftol': 1e-14})
                
                if result.success or result.fun < _objective_slsqp(x_slsqp0):
                    pos_sl = result.x[:2*N].reshape(N, 2)
                    radii_sl = np.maximum(result.x[2*N:], 1e-4)
                    pos_pp, radii_pp = _post_process(pos_sl, radii_sl)
                    score = np.sum(radii_pp)
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
                        
            except Exception:
                pass
    
    # QP radius optimization on best positions
    if time.time() < deadline - 10:
        try:
            pos_best = best_pos.copy()
            radii_qp = _qp_radius_optimize(pos_best, best_radii, max_iter=100, tol=1e-12)
            pos_pp, radii_pp = _post_process(pos_best, radii_qp)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final global polish on the best result found - high precision SLSQP
    if time.time() < deadline - 5:
        try:
            pos_final, radii_final = _post_process(best_pos, best_radii)
            x_final0 = np.concatenate([pos_final.flatten(), radii_final])
            cons_final = _slsqp_constraints(x_final0)
            
            # High-precision SLSQP polish
            result = minimize(_objective_slsqp, x_final0, method='SLSQP', bounds=bounds,
                              constraints=cons_final,
                              options={'maxiter': 500, 'ftol': 1e-16})
            
            if result.success:
                pos_pf = result.x[:2*N].reshape(N, 2)
                radii_pf = np.maximum(result.x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_pf, radii_pf)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
            
    # One more QP radius pass after final polish
    if time.time() < deadline - 3:
        try:
            radii_qp2 = _qp_radius_optimize(best_pos, best_radii, max_iter=100, tol=1e-12)
            pos_pp, radii_pp = _post_process(best_pos, radii_qp2)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to a known valid configuration
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result
>>>>>>> REPLACE
