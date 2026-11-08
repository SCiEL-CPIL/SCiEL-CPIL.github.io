<<<<<<< SEARCH
def _local_search(pos, radii, time_budget=10.0, start_time=None):
    """Local search: try small moves to improve packing."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() - start_time < time_budget:
        # Try to grow all radii slightly
        grow = 0.0005
        new_radii = radii + grow
        feasible = True
        for i in range(n):
            if new_radii[i] > 0.5:
                feasible = False
                break
            if pos[i, 0] < new_radii[i] - 1e-9 or pos[i, 0] > 1 - new_radii[i] + 1e-9:
                feasible = False
                break
            if pos[i, 1] < new_radii[i] - 1e-9 or pos[i, 1] > 1 - new_radii[i] + 1e-9:
                feasible = False
                break
        if feasible:
            for i, j in pairs:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < new_radii[i] + new_radii[j] - 1e-9:
                    feasible = False
                    break
        if feasible:
            radii = new_radii
            score = np.sum(radii)
            if score > best_score:
                best_score = score
            continue
        
        # Try small random moves
        for _ in range(100):
            if time.time() - start_time > time_budget:
                break
            idx = np.random.randint(n)
            old_pos = pos[idx].copy()
            pos[idx] += np.random.normal(0, 0.005, size=2)
            pos[idx] = np.clip(pos[idx], 0, 1)
            
            # Check validity
            valid = True
            if pos[idx, 0] < radii[idx] - 1e-9 or pos[idx, 0] > 1 - radii[idx] + 1e-9:
                valid = False
            if pos[idx, 1] < radii[idx] - 1e-9 or pos[idx, 1] > 1 - radii[idx] + 1e-9:
                valid = False
            if valid:
                for j in range(n):
                    if j == idx:
                        continue
                    dx = pos[j, 0] - pos[idx, 0]
                    dy = pos[j, 1] - pos[idx, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[idx] + radii[j] - 1e-9:
                        valid = False
                        break
            if not valid:
                pos[idx] = old_pos
    
    return pos, radii
=======
def _local_search(pos, radii, time_budget=10.0, start_time=None):
    """Local search: try small moves to improve packing."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    
    # Vectorized validity check helper
    def _is_valid(p, r):
        # Boundary
        if np.any(p[:, 0] < r - 1e-9) or np.any(p[:, 0] > 1 - r + 1e-9): return False
        if np.any(p[:, 1] < r - 1e-9) or np.any(p[:, 1] > 1 - r + 1e-9): return False
        # Pairwise
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = p[j_idx, 0] - p[i_idx, 0]
        dy = p[j_idx, 1] - p[i_idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[i_idx] + r[j_idx]
        if np.any(dist < min_dist - 1e-9): return False
        return True

    # Adaptive growth step
    grow_step = 0.001
    
    while time.time() - start_time < time_budget:
        # Try to grow all radii slightly
        new_radii = radii + grow_step
        
        # Quick check bounds
        if np.any(new_radii > 0.5):
            grow_step *= 0.5
            if grow_step < 1e-6: break
            continue
            
        # Check if growth is feasible with current positions
        feasible = True
        if np.any(pos[:, 0] < new_radii - 1e-9) or np.any(pos[:, 0] > 1 - new_radii + 1e-9):
            feasible = False
        elif np.any(pos[:, 1] < new_radii - 1e-9) or np.any(pos[:, 1] > 1 - new_radii + 1e-9):
            feasible = False
        else:
            # Pairwise check
            i_idx, j_idx = np.triu_indices(n, k=1)
            dx = pos[j_idx, 0] - pos[i_idx, 0]
            dy = pos[j_idx, 1] - pos[i_idx, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = new_radii[i_idx] + new_radii[j_idx]
            if np.any(dist < min_dist - 1e-9):
                feasible = False
        
        if feasible:
            radii = new_radii
            # Increase step size slightly if successful
            grow_step = min(grow_step * 1.1, 0.005)
        else:
            # Reduce step size
            grow_step *= 0.5
            if grow_step < 1e-6:
                # Try random jitter to escape local minima
                for _ in range(50):
                    if time.time() - start_time > time_budget: break
                    idx = np.random.randint(n)
                    old_pos = pos[idx].copy()
                    pos[idx] += np.random.normal(0, 0.01, size=2)
                    pos[idx] = np.clip(pos[idx], 0, 1)
                    if _is_valid(pos, radii):
                        # Keep the move
                        pass
                    else:
                        pos[idx] = old_pos
            continue
            
        # After growth, relax to ensure feasibility
        pos, _ = _relax_vectorized(pos, radii, max_iters=100, time_budget=0.5, start_time=time.time())
        
        if _is_valid(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
        else:
            # Rollback if invalid
            radii = radii - grow_step
            # Re-relax
            pos, _ = _relax_vectorized(pos, radii, max_iters=50, time_budget=0.2, start_time=time.time())
    
    return pos, radii
>>>>>>> REPLACE

