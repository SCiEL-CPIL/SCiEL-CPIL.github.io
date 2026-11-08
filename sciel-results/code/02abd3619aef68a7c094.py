<<<<<<< SEARCH
def _local_search(pos, radii, pairs, time_budget=20.0, start_time=None):
    """Local search with random moves and growth."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    
    while time.time() - start_time < time_budget:
        # Try to grow
        feasible, new_radii = _feasible_grow(pos, radii, 0.001, pairs)
        if feasible:
            radii = new_radii
            pos, radii = _relax(pos, radii, pairs, max_iters=100,
                                  time_budget=min(0.5, time_budget - (time.time() - start_time)),
                                  start_time=start_time)
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
            continue
        
        # Random move: pick a circle and nudge it
        for attempt in range(50):
            if time.time() - start_time > time_budget:
                break
            
            idx = rng.randint(n)
            old_pos = pos[idx].copy()
            step_size = 0.01
            pos[idx] += rng.normal(0, step_size, size=2)
            pos[idx] = np.clip(pos[idx], radii[idx], 1 - radii[idx])
            
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
            
            if valid:
                # Accept the move
                # Try to grow a bit
                feasible2, new_r = _feasible_grow(pos, radii, 0.0005, pairs)
                if feasible2:
                    radii = new_r
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
            else:
                pos[idx] = old_pos
    
    return best_pos, best_radii
=======
def _local_search(pos, radii, pairs, time_budget=20.0, start_time=None):
    """Local search with random moves and growth."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    
    while time.time() - start_time < time_budget:
        # Try to grow
        feasible, new_radii = _feasible_grow(pos, radii, 0.001, pairs)
        if feasible:
            radii = new_radii
            pos, radii = _relax(pos, radii, pairs, max_iters=100,
                                  time_budget=min(0.5, time_budget - (time.time() - start_time)),
                                  start_time=start_time)
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
            continue
        
        # Random move: pick a circle and nudge it
        for attempt in range(50):
            if time.time() - start_time > time_budget:
                break
            
            idx = rng.randint(n)
            old_pos = pos[idx].copy()
            step_size = 0.01
            pos[idx] += rng.normal(0, step_size, size=2)
            pos[idx] = np.clip(pos[idx], radii[idx], 1 - radii[idx])
            
            # Check validity using vectorized helper for speed
            # We can just use _feasible_grow with grow=0 to check validity? 
            # _feasible_grow checks if radii + 0 is feasible.
            valid, _ = _feasible_grow(pos, radii, 0.0, pairs)
            
            if valid:
                # Accept the move
                # Try to grow a bit
                feasible2, new_r = _feasible_grow(pos, radii, 0.0005, pairs)
                if feasible2:
                    radii = new_r
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
            else:
                pos[idx] = old_pos
    
    return best_pos, best_radii

def _polish_scipy(pos, radii, pairs, time_budget=10.0, start_time=None):
    """Polish using scipy minimize on a smooth penalty function."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    
    # Variables: x, y, r for each circle -> 3n variables
    x0 = np.concatenate([pos.flatten(), radii])
    
    def objective(v):
        pos_v = v[:2*n].reshape(n, 2)
        radii_v = v[2*n:]
        
        # Penalty for being out of bounds
        penalty = 0.0
        if np.any(pos_v[:, 0] < radii_v) or np.any(pos_v[:, 0] > 1 - radii_v):
            penalty += 100 * (np.sum(np.maximum(0, radii_v - pos_v[:, 0])) + 
                              np.sum(np.maximum(0, pos_v[:, 0] - (1 - radii_v))))
        if np.any(pos_v[:, 1] < radii_v) or np.any(pos_v[:, 1] > 1 - radii_v):
            penalty += 100 * (np.sum(np.maximum(0, radii_v - pos_v[:, 1])) + 
                              np.sum(np.maximum(0, pos_v[:, 1] - (1 - radii_v))))
            
        # Penalty for overlap
        diff = pos_v[:, np.newaxis, :] - pos_v[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        req = radii_v[:, np.newaxis] + radii_v[np.newaxis, :]
        overlap = np.maximum(0, req - dist)
        # Sum of overlaps
        overlap_penalty = np.sum(overlap)
        
        # We want to maximize sum(radii), so minimize -sum(radii) + penalty
        return -np.sum(radii_v) + 10 * overlap_penalty + penalty

    # Bounds:
    # x, y in [0, 1] (though effectively constrained by r)
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-12})
        
        if res.success:
            best_pos = res.x[:2*n].reshape(n, 2)
            best_radii = res.x[2*n:]
            
            # Validate and clamp
            _clamp_boundaries(best_pos, best_radii)
            if _check_validity(best_pos, best_radii):
                return best_pos, best_radii
    except Exception:
        pass
        
    return pos, radii
>>>>>>> REPLACE

