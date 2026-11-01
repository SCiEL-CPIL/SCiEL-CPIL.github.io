<<<<<<< SEARCH
    # Optimize each start, allocating time budget
    best_pts = None
    best_energy = np.inf
    
    n_starts = len(starts)
    time_per_start = (DEADLINE_S - (time.time() - t0)) / n_starts
    # Reserve 20% for polishing
    time_per_opt = time_per_start * 0.8
    polish_time = time_per_start * 0.2
    
    for i, start in enumerate(starts):
        elapsed = time.time() - t0
        remaining = DEADLINE_S - elapsed
        
        if remaining < 5.0:
            break
        
        # Allocate remaining time fairly
        time_left_for_opt = min(time_per_opt, remaining * 0.7 / (n_starts - i))
        time_left_for_opt = max(time_left_for_opt, 1.0)
        
        pts, energy = _optimize_from_start(start, time_left_for_opt)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final polish on the best result
    elapsed = time.time() - t0
    remaining = DEADLINE_S - elapsed
    
    if best_pts is not None and remaining > 3.0:
        polish_budget = min(remaining * 0.8, 30.0)
        best_pts, best_energy = _polish(best_pts, polish_budget)
=======
    # Optimize each start using both methods, allocating time budget
    best_pts = None
    best_energy = np.inf
    
    n_starts = len(starts)
    time_per_start = (DEADLINE_S - (time.time() - t0)) / n_starts
    # Reserve 25% for polishing
    time_per_opt = time_per_start * 0.75
    polish_time = time_per_start * 0.25
    
    for i, start in enumerate(starts):
        elapsed = time.time() - t0
        remaining = DEADLINE_S - elapsed
        
        if remaining < 5.0:
            break
        
        # Allocate remaining time fairly
        time_left_for_opt = min(time_per_opt, remaining * 0.7 / (n_starts - i))
        time_left_for_opt = max(time_left_for_opt, 1.0)
        
        # Use Riemannian CG as primary optimizer (better for manifold)
        pts, energy = _riemannian_cg(start, time_left_for_opt * 0.6, tol=1e-10)
        
        # Also try projected gradient descent if time allows
        if time_left_for_opt > 3.0:
            pts2, energy2 = _optimize_from_start(start, time_left_for_opt * 0.4)
            if energy2 < energy:
                pts, energy = pts2, energy2
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final polish on the best result
    elapsed = time.time() - t0
    remaining = DEADLINE_S - elapsed
    
    if best_pts is not None and remaining > 3.0:
        polish_budget = min(remaining * 0.8, 40.0)
        best_pts, best_energy = _polish(best_pts, polish_budget)
>>>>>>> REPLACE
