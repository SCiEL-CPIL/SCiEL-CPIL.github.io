<<<<<<< SEARCH
def _objective(params, beta=10.0):
    """
    Objective function: Soft max of facet radii.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    facets, circumcenters, radii = _compute_hull_data(X)
    if facets is None:
        return 1e10
        
    # Soft max: (1/beta) * log(sum(exp(beta * r)))
    # To improve stability, shift by max
    r_max = np.max(radii)
    if r_max == 0:
        return 0.0
        
    shifted = beta * (radii - r_max)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
    
    return log_sum_exp

def _repulsion_relax(points, alpha_init=0.05, decay=0.995, max_iter=300, deadline=None):
    """
    Repulsion Relaxation (Thomson Problem solver).
    Minimize sum 1/||p_i - p_j||^2 to spread points evenly.
    """
    points = points.copy()
    alpha = alpha_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Calculate forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        # Force F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project onto tangent plane: F_tan = F - (F . p) p
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        # Update
        points = points + alpha * forces_tan
        
        # Normalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Decay alpha
        alpha *= decay
        
    return points

def _lloyd_relax(points, steps=100, alpha=0.5, deadline=None):
    """
    Lloyd's algorithm for sphere covering.
    Moves each point towards the centroid of its Voronoi cell (approximated by incident facet circumcenters).
    """
    points = points.copy()
    for _ in range(steps):
        if deadline is not None and time.time() > deadline:
            break
            
        facets, circumcenters, radii = _compute_hull_data(points)
        if facets is None:
            break
            
        # Map point index to list of incident facet indices
        # facets is (F, 3) array of point indices
        # We need to know for each point i, which facets contain it.
        # Since N=50, we can just loop or use bincount logic.
        
        # Create a matrix M of shape (N, F) where M[i, f] = 1 if i is in facet f
        # But facets are triples.
        # Let's just collect circumcenters for each point.
        
        new_points = points.copy()
        
        # Efficient way: 
        # For each facet f, it has points p0, p1, p2 and circumcenter cc_f.
        # We want to sum cc_f for all f incident to point i.
        
        # Initialize sums
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        for f in range(len(facets)):
            cc = circumcenters[f]
            # If circumcenter is zero (degenerate), skip or handle
            if np.linalg.norm(cc) < 1e-9:
                continue
            i0, i1, i2 = facets[f]
            sums[i0] += cc
            counts[i0] += 1
            sums[i1] += cc
            counts[i1] += 1
            sums[i2] += cc
            counts[i2] += 1
        
        # Compute centroids
        valid = counts > 0
        centroids = points[valid].copy()
        centroids[valid] = sums[valid] / counts[valid, np.newaxis]
        
        # Normalize centroids to unit sphere
        norms = np.linalg.norm(centroids, axis=1)
        norms[norms < 1e-9] = 1.0
        centroids = centroids / norms[:, np.newaxis]
        
        # Update points
        # new_points[valid] = (1-alpha)*points[valid] + alpha*centroids
        # Then renormalize
        update = points[valid] * (1 - alpha) + centroids * alpha
        update_norms = np.linalg.norm(update, axis=1)
        update_norms[update_norms < 1e-9] = 1.0
        update = update / update_norms[:, np.newaxis]
        
        new_points[valid] = update
        
        points = new_points
        
    return points

def _local_optimize(config, deadline):
    """
    Local optimization:
    1. Lloyd's relaxation
    2. L-BFGS on soft-max objective
    3. Lloyd's relaxation again
    4. Final L-BFGS polish
    """
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(best_config)
    
    # Phase 1: Lloyd's Relaxation
    if time.time() < deadline - 10.0:
        lloyd_config = _lloyd_relax(best_config, steps=50, alpha=0.5, deadline=deadline - 5.0)
        lloyd_score = _covering_radius_deg_hull(lloyd_config)
        if lloyd_score < best_score:
            best_score = lloyd_score
            best_config = lloyd_config.copy()
            
    # Phase 2: L-BFGS with increasing beta
    betas = [10.0, 50.0, 200.0]
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        
        try:
            res = minimize(_objective, best_config.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100},
                           args=(beta,))
            
            new_config = res.x.reshape(N, 3)
            norms = np.linalg.norm(new_config, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            new_config = new_config / norms
            
            new_score = _covering_radius_deg_hull(new_config)
            
            if new_score < best_score:
                best_score = new_score
                best_config = new_config.copy()
                
        except Exception:
            pass
        
    # Phase 3: Final Lloyd Polish
    if time.time() < deadline - 2.0:
        lloyd_config = _lloyd_relax(best_config, steps=20, alpha=0.2, deadline=deadline - 1.0)
        lloyd_score = _covering_radius_deg_hull(lloyd_config)
        if lloyd_score < best_score:
            best_score = lloyd_score
            best_config = lloyd_config.copy()
            
    # Phase 4: Very high beta L-BFGS
    if time.time() < deadline - 0.5:
        try:
            res = minimize(_objective, best_config.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50},
                           args=(500.0,))
            
            new_config = res.x.reshape(N, 3)
            norms = np.linalg.norm(new_config, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            new_config = new_config / norms
            
            new_score = _covering_radius_deg_hull(new_config)
            
            if new_score < best_score:
                best_score = new_score
                best_config = new_config.copy()
                
        except Exception:
            pass
            
    return best_config, best_score
=======
def _lloyd_relax(points, steps=10, alpha=0.5, deadline=None):
    """
    Lloyd's algorithm for sphere covering.
    """
    points = points.copy()
    for _ in range(steps):
        if deadline is not None and time.time() > deadline:
            break
            
        facets, circumcenters, radii = _compute_hull_data(points)
        if facets is None:
            break
            
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        for f in range(len(facets)):
            cc = circumcenters[f]
            if np.linalg.norm(cc) < 1e-9:
                continue
            i0, i1, i2 = facets[f]
            sums[i0] += cc
            counts[i0] += 1
            sums[i1] += cc
            counts[i1] += 1
            sums[i2] += cc
            counts[i2] += 1
        
        valid = counts > 0
        centroids = points[valid].copy()
        centroids[valid] = sums[valid] / counts[valid, np.newaxis]
        
        norms = np.linalg.norm(centroids, axis=1)
        norms[norms < 1e-9] = 1.0
        centroids = centroids / norms[:, np.newaxis]
        
        update = points[valid] * (1 - alpha) + centroids * alpha
        update_norms = np.linalg.norm(update, axis=1)
        update_norms[update_norms < 1e-9] = 1.0
        update = update / update_norms[:, np.newaxis]
        
        points[valid] = update
        
    return points

def _simulated_annealing(config, deadline):
    """
    Simulated Annealing with periodic Lloyd relaxation.
    """
    current_config = config.copy()
    current_score = _covering_radius_deg_hull(current_config)
    best_config = current_config.copy()
    best_score = current_score
    
    T_start = 5.0
    T_end = 0.1
    T = T_start
    decay = 0.99
    
    while time.time() < deadline - 5.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            
            idx = np.random.randint(N)
            noise = np.random.randn(3) * 0.5
            new_p = current_config[idx] + noise
            new_p = _normalize(new_p)
            
            old_p = current_config[idx].copy()
            current_config[idx] = new_p
            
            new_score = _covering_radius_deg_hull(current_config)
            delta = new_score - current_score
            
            if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_config = current_config.copy()
            else:
                current_config[idx] = old_p
        
        T *= decay
        
        if time.time() < deadline - 5.0:
            relaxed_config = _lloyd_relax(current_config, steps=5, alpha=0.5, deadline=deadline - 5.0)
            relaxed_score = _covering_radius_deg_hull(relaxed_config)
            
            if relaxed_score < current_score:
                current_config = relaxed_config
                current_score = relaxed_score
                if relaxed_score < best_score:
                    best_score = relaxed_score
                    best_config = relaxed_config.copy()
            
    return best_config, best_score

def _lbfgs_polish(config, deadline):
    """
    Final polish with L-BFGS on soft-max objective.
    """
    def _objective(params, beta=500.0):
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
        
        facets, circumcenters, radii = _compute_hull_data(X)
        if facets is None:
            return 1e10
            
        r_max = np.max(radii)
        if r_max == 0:
            return 0.0
            
        shifted = beta * (radii - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    try:
        res = minimize(_objective, config.flatten(), method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100},
                       args=(500.0,))
        
        new_config = res.x.reshape(N, 3)
        norms = np.linalg.norm(new_config, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        new_config = new_config / norms
        
        new_score = _covering_radius_deg_hull(new_config)
        
        return new_config, new_score
    except Exception:
        return config, _covering_radius_deg_hull(config)
>>>>>>> REPLACE

And then replace `construct`:

