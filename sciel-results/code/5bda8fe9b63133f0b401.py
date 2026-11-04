<<<<<<< SEARCH
def optimize_configuration(pos, angle_deg=0.0, max_iter=3000):
    """Optimize a configuration using force-directed relaxation and local search."""
    pos, _ = force_directed_relaxation(pos, angle_deg, max_iter=max_iter)
    
    # Local search: try small perturbations with adaptive step size
    n = len(pos)
    step_size = 0.05
    current_obj = float('inf')
    
    # Compute initial objective
    best_S = float('inf')
    for a in np.linspace(0, 60, 13):
        S, _, _ = min_enclosing_hexagon_side_fast(pos, a)
        if S < best_S:
            best_S = S
    current_obj = best_S
    
    no_improve_count = 0
    for it in range(50):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        improved = False
        for i in range(n):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    
                    # Check if valid (no overlap)
                    if not check_no_overlap(new_pos, angle_deg, tol=1e-3):
                        continue
                    
                    # Compute objective
                    best_local_S = float('inf')
                    for a in np.linspace(0, 60, 7):
                        S, _, _ = min_enclosing_hexagon_side_fast(new_pos, a)
                        if S < best_local_S:
                            best_local_S = S
                    
                    if best_local_S < current_obj - 1e-6:
                        pos = new_pos
                        current_obj = best_local_S
                        improved = True
        
        if not improved:
            step_size *= 0.7
            no_improve_count += 1
            if no_improve_count > 3 or step_size < 0.003:
                break
        else:
            no_improve_count = 0
    
    return pos, current_obj
=======
def optimize_configuration(pos, angle_deg=0.0, max_iter=3000):
    """Optimize a configuration using force-directed relaxation and local search."""
    pos, _ = force_directed_relaxation(pos, angle_deg, max_iter=max_iter)
    
    # Local search: try small perturbations with adaptive step size
    n = len(pos)
    step_size = 0.08
    current_obj = float('inf')
    
    # Compute initial objective
    best_S = float('inf')
    for a in np.linspace(0, 60, 13):
        S, _, _ = min_enclosing_hexagon_side_fast(pos, a)
        if S < best_S:
            best_S = S
    current_obj = best_S
    
    no_improve_count = 0
    for it in range(80):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        improved = False
        for i in range(n):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            # Try 8 directions + diagonal moves
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    
                    # Check if valid (no overlap)
                    if not check_no_overlap(new_pos, angle_deg, tol=1e-3):
                        continue
                    
                    # Compute objective
                    best_local_S = float('inf')
                    for a in np.linspace(0, 60, 7):
                        S, _, _ = min_enclosing_hexagon_side_fast(new_pos, a)
                        if S < best_local_S:
                            best_local_S = S
                    
                    if best_local_S < current_obj - 1e-6:
                        pos = new_pos
                        current_obj = best_local_S
                        improved = True
        
        if not improved:
            step_size *= 0.75
            no_improve_count += 1
            if no_improve_count > 4 or step_size < 0.002:
                break
        else:
            no_improve_count = 0
    
    return pos, current_obj
>>>>>>> REPLACE

