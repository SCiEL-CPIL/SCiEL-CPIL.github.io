<<<<<<< SEARCH
def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(300):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p
=======
def _resolve_constraints(params, max_iter=1000):
    """
    Robustly enforce all constraints by iteratively pushing circles apart
    and clamping to boundaries. This guarantees a valid output if it converges.
    """
    p = params.reshape(N, 3).copy()
    
    # Initial clipping
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Ensure radii fit in walls initially
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    for _ in range(max_iter):
        # 1. Check and fix boundary violations
        # If a circle is too close to a wall, move it away and/or shrink it
        for i in range(N):
            # X-bound
            if p[i, 0] < p[i, 2]:
                p[i, 0] = p[i, 2]
            elif p[i, 0] > 1.0 - p[i, 2]:
                p[i, 0] = 1.0 - p[i, 2]
            
            # Y-bound
            if p[i, 1] < p[i, 2]:
                p[i, 1] = p[i, 2]
            elif p[i, 1] > 1.0 - p[i, 2]:
                p[i, 1] = 1.0 - p[i, 2]
            
            # After moving, check if radius still fits (should, but safe)
            wall = np.minimum(np.minimum(p[i, 0], 1.0 - p[i, 0]),
                              np.minimum(p[i, 1], 1.0 - p[i, 1]))
            if p[i, 2] > wall:
                p[i, 2] = max(wall, 1e-6)

        # 2. Check and fix overlaps
        overlaps_found = False
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = p[iu[0], 2] + p[iu[1], 2]
        overlap = min_dist - dist
        
        # Use a small tolerance for "active" overlap
        active = overlap > 1e-9
        
        if not np.any(active):
            break
            
        overlaps_found = True
        
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            d = dist[k]
            
            if d < 1e-9:
                # If centers coincide, push apart arbitrarily
                dx_dir = 1.0
                dy_dir = 0.0
            else:
                dx_dir = dx[k] / d
                dy_dir = dy[k] / d
            
            # Push apart by half the overlap each
            push = ov * 0.5 + 1e-9
            
            p[i, 0] += dx_dir * push
            p[i, 1] += dy_dir * push
            p[j, 0] -= dx_dir * push
            p[j, 1] -= dy_dir * push
            
            # Re-clamp to boundaries immediately after push
            # This might create new overlaps, but the loop handles it
            p[i, 0] = np.clip(p[i, 0], p[i, 2], 1.0 - p[i, 2])
            p[i, 1] = np.clip(p[i, 1], p[i, 2], 1.0 - p[i, 2])
            p[j, 0] = np.clip(p[j, 0], p[j, 2], 1.0 - p[j, 2])
            p[j, 1] = np.clip(p[j, 1], p[j, 2], 1.0 - p[j, 2])
            
            # If clamping caused radius to exceed wall (shouldn't happen with clip logic above, 
            # but just in case the clip logic was wrong in previous step)
            wall_i = np.minimum(np.minimum(p[i, 0], 1.0 - p[i, 0]),
                                np.minimum(p[i, 1], 1.0 - p[i, 1]))
            if p[i, 2] > wall_i:
                p[i, 2] = max(wall_i, 1e-6)
                
            wall_j = np.minimum(np.minimum(p[j, 0], 1.0 - p[j, 0]),
                                np.minimum(p[j, 1], 1.0 - p[j, 1]))
            if p[j, 2] > wall_j:
                p[j, 2] = max(wall_j, 1e-6)

    # Final safety check: if still overlapping, shrink radii drastically
    # This is a fallback for pathological cases
    if not _validate(p):
        for i in range(N):
            p[i, 2] *= 0.9
        p = _resolve_constraints(p, max_iter=500)

    return p
>>>>>>> REPLACE

