# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _ring_initialization(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    # Target radius r ~ 1/sqrt(n) ~ 0.057
    # Spacing ~ 2r ~ 0.114
    # Let's try a range of spacings to ensure we get enough points
    
    best_points = None
    for scale in np.linspace(0.9, 1.1, 20):
        spacing = 2.0 / (np.sqrt(n) * scale)
        
        # Generate hex lattice
        # Radius of lattice in terms of grid index
        max_k = int(1.0 / spacing) + 2
        
        rows = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append([x, y])
        
        if len(rows) >= n:
            points = np.array(rows)
            # Sort by radius to keep central points
            radii = np.linalg.norm(points, axis=1)
            points = points[np.argsort(radii, kind="stable")[:n]]
            # Add tiny random noise to break symmetry? 
            # This can help L-BFGS escape symmetric local minima.
            points += np.random.normal(0, 1e-4, size=points.shape)
            return points
            
    # Fallback: uniform random
    return np.random.uniform(-0.9, 0.9, size=(n, 2))

def _soft_min_objective_and_grad(x, beta=50.0):
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists_upper = dists[mask]
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Soft min calculation
    exp_pairs = -beta * dists_upper / 2.0
    exp_bound = -beta * boundary_dists
    
    all_exps = np.concatenate([exp_pairs, exp_bound])
    max_exp = np.max(all_exps)
    
    shifted_pairs = exp_pairs - max_exp
    shifted_bound = exp_bound - max_exp
    
    sum_pairs = np.sum(np.exp(shifted_pairs))
    sum_bound = np.sum(np.exp(shifted_bound))
    
    S_log = max_exp + np.log(sum_pairs + sum_bound)
    
    objective = S_log / beta
    
    # Gradient
    denom = sum_pairs + sum_bound
    if denom < 1e-300:
        return objective, np.zeros_like(x)
        
    # Weights
    w_pairs = np.exp(shifted_pairs) / denom
    
    # Full weight matrix
    W = np.zeros((n, n))
    W[mask] = w_pairs
    
    # Gradient of pairs
    dists_safe = dists.copy()
    np.fill_diagonal(dists_safe, 1.0)
    
    # U[i, j] = (p_i - p_j) / (2 * d_ij)
    U = diff / (2.0 * dists_safe[:, :, np.newaxis])
    
    grad_pair = np.sum(W[:, :, np.newaxis] * U, axis=1)
    
    # Gradient of boundaries
    w_bound = np.exp(shifted_bound) / denom
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    grad_bound = -w_bound[:, np.newaxis] * (p / norms_safe[:, np.newaxis])
    
    grad_m = grad_pair + grad_bound
    
    # Minimize -m(x)
    return objective, -grad_m.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization
    points = _ring_initialization(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Soft-min optimization
    try:
        # Use L-BFGS-B
        res = minimize(
            lambda x: _soft_min_objective_and_grad(x, beta=50.0)[0],
            points.flatten(),
            jac=lambda x: _soft_min_objective_and_grad(x, beta=50.0)[1],
            method='L-BFGS-B',
            options={
                'maxiter': 200,
                'gtol': 1e-6,
                'ftol': 1e-10,
                'maxls': 20,
                'disp': False
            }
        )
        
        if time.time() < deadline:
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                
        # 3. Hard optimization (polish)
        # Use the original subgradient logic or a tighter soft-min
        if time.time() < deadline - 10:
            # Try a higher beta for closer approximation to hard min
            res2 = minimize(
                lambda x: _soft_min_objective_and_grad(x, beta=100.0)[0],
                best_points.flatten(),
                jac=lambda x: _soft_min_objective_and_grad(x, beta=100.0)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 20,
                    'disp': False
                }
            )
            if time.time() < deadline:
                candidate2 = res2.x.reshape(-1, 2)
                candidate_score2 = _compute_score(candidate2)
                if candidate_score2 > best_score:
                    best_points = candidate2
                    best_score = candidate_score2
                    
    except Exception as e:
        pass

    # Final check
    norms = np.linalg.norm(best_points, axis=1)
    # Ensure points are inside
    if np.any(norms > 1.0 - 1e-9):
        # Scale back slightly
        scale = (1.0 - 1e-9) / np.max(norms)
        best_points *= scale
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
