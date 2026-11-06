<<<<<<< SEARCH
def _repulsion_relax(pts, deadline, seed=123):
    """Repulsion-based relaxation to untangle close pairs."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Iterative repulsion: push apart points that are too close
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Find the closest pair
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        if min_dist > 0.106:  # Above target spacing
            break
        
        # Find indices of closest pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        if i == j:
            continue
        
        # Push them apart
        mid = (current[i] + current[j]) / 2.0
        direction = current[i] - current[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = rng.randn(2)
            norm = np.linalg.norm(direction)
        direction = direction / norm
        
        push = 0.002
        current[i] += direction * push
        current[j] -= direction * push
        
        # Keep inside circle
        for k in [i, j]:
            r_mag = np.hypot(current[k, 0], current[k, 1])
            if r_mag > 0.95:
                current[k] = current[k] / r_mag * 0.95
        
        new_r = _compute_radius(current)
        if new_r > current_r + 1e-10:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
    
    return best, best_r
=======
def _force_relax(pts, deadline, seed=123):
    """Force-based relaxation: compute net forces from overlaps and boundary violations."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Target spacing based on current radius
    target_spacing = max(current_r * 2.0, 0.01)
    
    for outer_iter in range(5000):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(current)
        
        # Overlap repulsion: for each pair, if distance < target_spacing, push apart
        for i in range(N):
            for j in range(i + 1, N):
                diff = current[i] - current[j]
                dist = np.sqrt(np.dot(diff, diff))
                if dist < target_spacing and dist > 1e-10:
                    # Repulsive force proportional to overlap
                    overlap = target_spacing - dist
                    force_mag = overlap * 0.5
                    direction = diff / dist
                    forces[i] += direction * force_mag
                    forces[j] -= direction * force_mag
                elif dist <= 1e-10:
                    # Coincident points: random push
                    direction = rng.randn(2)
                    direction /= np.linalg.norm(direction)
                    forces[i] += direction * 0.01
                    forces[j] -= direction * 0.01
        
        # Boundary force: pull inward if too close to boundary
        for i in range(N):
            r_mag = np.hypot(current[i, 0], current[i, 1])
            if r_mag > 0.95:
                # Inward force proportional to how far outside safe zone
                overshoot = r_mag - 0.95
                direction = -current[i] / r_mag
                forces[i] += direction * overshoot * 10.0
        
        # Apply forces with damping
        step = 0.01
        for i in range(N):
            current[i] += forces[i] * step
        
        # Keep inside circle (hard constraint)
        for i in range(N):
            r_mag = np.hypot(current[i, 0], current[i, 1])
            if r_mag > 0.99:
                current[i] = current[i] / r_mag * 0.99
        
        new_r = _compute_radius(current)
        if new_r > current_r + 1e-12:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
            # Decrease step size as we converge
            step *= 0.995
        else:
            # Revert if no improvement
            # Recompute from best
            current = best.copy()
            step *= 0.9
            if step < 0.0001:
                break
    
    return best, best_r
>>>>>>> REPLACE

