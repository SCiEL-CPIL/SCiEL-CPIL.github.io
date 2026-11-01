<<<<<<< SEARCH
    # Final validation and normalization
    if best_pts is None:
        best_pts = inits[0]
    
    best_pts = _retract(best_pts)
    
    # Ensure no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to spiral if points are too close
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _retract(best_pts)
    
    return best_pts
=======
    # Final validation and normalization
    if best_pts is None:
        best_pts = inits[0]
    
    best_pts = _retract(best_pts)
    
    # One more Newton + ultra polish pass on the best solution with remaining time
    elapsed = time.time() - start_time
    if elapsed < total_budget - 3.0:
        final_deadline = deadline - 0.5
        best_pts = _newton_polish(best_pts, final_deadline)
        best_pts = _ultra_polish(best_pts, final_deadline)
    
    # Ensure no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to spiral if points are too close
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _retract(best_pts)
    
    # Final energy check
    f_final, _ = _energy_and_grad(best_pts)
    if f_final < best_energy:
        best_energy = f_final
    
    return best_pts
>>>>>>> REPLACE
