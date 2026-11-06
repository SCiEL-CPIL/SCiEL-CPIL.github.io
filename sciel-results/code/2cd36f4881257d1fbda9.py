<<<<<<< SEARCH
def _gradient_ascent(pts, deadline, seed=456):
    """Gradient ascent on a smooth surrogate objective."""
    from scipy.optimize import minimize
    
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Define smooth surrogate using log-sum-exp
    def objective(x):
        x = x.reshape(-1, 2)
        # Boundary term
        dists_origin = np.hypot(x[:, 0], x[:, 1])
        boundary_term = np.max(dists_origin)
        
        # Pairwise distances - use soft min via log-sum-exp
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Soft min of distances
        beta = 50.0
        soft_min_dist = -np.log(np.sum(np.exp(-beta * dists_sq))) / beta
        
        # Objective: minimize negative radius
        # radius = min(min_pair/2, 1 - max_dist)
        neg_radius = -min(soft_min_dist / 2.0, 1.0 - boundary_term)
        return neg_radius
    
    def objective_grad(x):
        x = x.reshape(-1, 2)
        dists_origin = np.hypot(x[:, 0], x[:, 1])
        boundary_term = np.max(dists_origin)
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        beta = 50.0
        weights = np.exp(-beta * dists_sq)
        weights /= np.sum(weights)
        
        # Vectorized gradient of soft min
        # grad[i] = sum_j weights[i,j] * 2*(x[i]-x[j])
        # = 2 * (x[i] * sum_j weights[i,j] - sum_j weights[i,j]*x[j])
        row_sums = np.sum(weights, axis=1, keepdims=True)
        weighted_sum = weights @ x  # shape (N, 2)
        grad = 2.0 * (x * row_sums - weighted_sum)
        
        # Gradient of boundary
        max_idx = np.argmax(dists_origin)
        boundary_grad = np.zeros_like(x)
        boundary_grad[max_idx] = x[max_idx] / dists_origin[max_idx]
        
        # Combine
        soft_min_dist = -np.log(np.sum(np.exp(-beta * dists_sq))) / beta
        
        if soft_min_dist / 2.0 < 1.0 - boundary_term:
            total_grad = grad / 2.0
        else:
            total_grad = boundary_grad
        
        return total_grad.flatten()
    
    # Run a few iterations of gradient ascent
    x = current.flatten()
    for _ in range(50):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(objective, x, jac=objective_grad, method='L-BFGS-B',
                             options={'maxiter': 30, 'gtol': 1e-5})
            x_new = result.x.reshape(-1, 2)
            
            # Check if improvement
            new_r = _compute_radius_fast(x_new)
            if new_r > current_r + 1e-12:
                current = x_new
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best = current.copy()
        except Exception:
            break
    
    return best, best_r
=======
def _gradient_ascent(pts, deadline, seed=456):
    """Gradient ascent on a smooth surrogate objective with restarts."""
    from scipy.optimize import minimize
    
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Define smooth surrogate using log-sum-exp
    def objective(x, beta=30.0):
        x = x.reshape(-1, 2)
        # Boundary term
        dists_origin = np.hypot(x[:, 0], x[:, 1])
        boundary_term = np.max(dists_origin)
        
        # Pairwise distances - use soft min via log-sum-exp
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Soft min of distances
        soft_min_dist_sq = -np.log(np.sum(np.exp(-beta * dists_sq))) / beta
        soft_min_dist = np.sqrt(soft_min_dist_sq)
        
        # Objective: minimize negative radius
        # radius = min(min_pair/2, 1 - max_dist)
        neg_radius = -min(soft_min_dist / 2.0, 1.0 - boundary_term)
        return neg_radius
    
    def objective_grad(x, beta=30.0):
        x = x.reshape(-1, 2)
        dists_origin = np.hypot(x[:, 0], x[:, 1])
        boundary_term = np.max(dists_origin)
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Soft min gradient
        weights = np.exp(-beta * dists_sq)
        weights /= np.sum(weights)
        
        # grad of soft_min_dist_sq[i] = sum_j weights[i,j] * 2*(x[i]-x[j])
        row_sums = np.sum(weights, axis=1, keepdims=True)
        weighted_sum = weights @ x  # shape (N, 2)
        grad_sq = 2.0 * (x * row_sums - weighted_sum)
        
        # grad of soft_min_dist = grad_sq / (2 * soft_min_dist)
        soft_min_dist_sq = -np.log(np.sum(np.exp(-beta * dists_sq))) / beta
        soft_min_dist = np.sqrt(soft_min_dist_sq)
        
        # Avoid division by zero
        safe_dist = np.where(soft_min_dist > 1e-10, soft_min_dist, 1.0)
        grad_dist = grad_sq / (2.0 * safe_dist)
        
        # Gradient of boundary
        max_idx = np.argmax(dists_origin)
        boundary_grad = np.zeros_like(x)
        boundary_grad[max_idx] = x[max_idx] / dists_origin[max_idx]
        
        # Combine: use the active constraint
        if soft_min_dist / 2.0 < 1.0 - boundary_term:
            total_grad = grad_dist / 2.0
        else:
            total_grad = boundary_grad
        
        return total_grad.flatten()
    
    # Run gradient ascent with multiple beta values and restarts
    betas = [20.0, 30.0, 50.0]
    for beta in betas:
        if time.time() > deadline:
            break
        
        x = current.flatten()
        for restart in range(3):
            if time.time() > deadline:
                break
            
            try:
                result = minimize(lambda v: objective(v, beta), x, 
                                 jac=lambda v: objective_grad(v, beta),
                                 method='L-BFGS-B',
                                 options={'maxiter': 50, 'gtol': 1e-6})
                x_new = result.x.reshape(-1, 2)
                
                # Check if improvement
                new_r = _compute_radius_fast(x_new)
                if new_r > current_r + 1e-12:
                    current = x_new
                    current_r = new_r
                    x = x_new.flatten()
                    if new_r > best_r:
                        best_r = new_r
                        best = current.copy()
                else:
                    # Small perturbation for restart
                    x = x + rng.randn(2 * N) * 0.001
            except Exception:
                break
    
    return best, best_r
>>>>>>> REPLACE

