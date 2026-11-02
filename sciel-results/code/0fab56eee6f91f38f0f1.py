<<<<<<< SEARCH
def _slsqp_polish(c, r_target=None):
    """
    SLSQP polish with active-set constraints.
    Maximizes r subject to pairwise and boundary constraints.
    """
    if r_target is None:
        r_target = _compute_radius(c) * 0.999
    
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    # Objective: minimize -r (maximize r)
    # We use a smooth approximation: minimize the negative of the soft-min
    def objective(params):
        c_p = params.reshape(N, 2)
        x = c_p[:, 0]
        y = c_p[:, 1]
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1.0)
        
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        d_bounds = np.concatenate([d_left, d_right, d_down, d_up])
        
        all_d = np.concatenate([d_pairs, d_bounds])
        # Minimize negative soft-max (equivalent to maximizing soft-min)
        beta = 500.0
        d_max = np.max(all_d)
        shifted = d_max - all_d
        weights = np.exp(-beta * shifted)
        sum_w = np.sum(weights)
        soft_max = d_max + np.log(sum_w) / beta
        return -soft_max
    
    def objective_grad(params):
        c_p = params.reshape(N, 2)
        x = c_p[:, 0]
        y = c_p[:, 1]
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1.0)
        
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        d_bounds = np.concatenate([d_left, d_right, d_down, d_up])
        
        all_d = np.concatenate([d_pairs, d_bounds])
        beta = 500.0
        d_max = np.max(all_d)
        shifted = d_max - all_d
        weights = np.exp(-beta * shifted)
        sum_w = np.sum(weights)
        p = weights / sum_w
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        i_idx, j_idx = iu
        vec_x = x[i_idx] - x[j_idx]
        vec_y = y[i_idx] - y[j_idx]
        inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
        
        p_pairs = p[:len(d_pairs)]
        
        # For soft-max, gradient sign is flipped
        grad_x[i_idx] += p_pairs * vec_x * inv_d
        grad_x[j_idx] -= p_pairs * vec_x * inv_d
        grad_y[i_idx] += p_pairs * vec_y * inv_d
        grad_y[j_idx] -= p_pairs * vec_y * inv_d
        
        p_bounds = p[len(d_pairs):]
        p_left = p_bounds[:N]
        p_right = p_bounds[N:2*N]
        p_down = p_bounds[2*N:3*N]
        p_up = p_bounds[3*N:]
        
        grad_x += p_left - p_right
        grad_y += p_down - p_up
        
        total_grad = np.concatenate([grad_x, grad_y])
        return -total_grad  # Negate because we're minimizing -soft_max
    
    # Constraints: all pairwise distances >= 2*r_target, boundary distances >= r_target
    constraints = []
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    # Only add constraints for pairs that are close to the target
    # This keeps the constraint set manageable
    c_flat = c.flatten()
    x = c[:, 0]
    y = c[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    d_pairs = d[iu]
    
    # Add constraints for pairs within 1.2 * 2*r_target
    close_pairs = np.where(d_pairs < 1.2 * 2 * r_target)[0]
    for idx in close_pairs[:200]:  # Limit number of constraints
        ii = i_idx[idx]
        jj = j_idx[idx]
        def pair_constraint(params, ii=ii, jj=jj):
            c_p = params.reshape(N, 2)
            return np.linalg.norm(c_p[ii] - c_p[jj]) - 2 * r_target
        constraints.append({'type': 'ineq', 'fun': pair_constraint})
    
    # Boundary constraints
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    close_bounds = np.where(np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up)) < 1.2 * r_target)[0]
    for i in close_bounds:
        def bound_constraint(params, i=i):
            c_p = params.reshape(N, 2)
            return np.min([0.5 - c_p[i, 0], 0.5 + c_p[i, 0], 0.5 - c_p[i, 1], 0.5 + c_p[i, 1]]) - r_target
        constraints.append({'type': 'ineq', 'fun': bound_constraint})
    
    try:
        res = minimize(
            objective,
            c.flatten(),
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c
=======
def _slsqp_polish(c, r_target=None, max_iter=300):
    """
    SLSQP polish with active-set constraints.
    Maximizes r subject to pairwise and boundary constraints.
    """
    if r_target is None:
        r_target = _compute_radius(c) * 0.999
    
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    # Objective: minimize -r (maximize r)
    # We use a smooth approximation: minimize the negative of the soft-min
    def objective(params):
        c_p = params.reshape(N, 2)
        x = c_p[:, 0]
        y = c_p[:, 1]
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1.0)
        
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        d_bounds = np.concatenate([d_left, d_right, d_down, d_up])
        
        all_d = np.concatenate([d_pairs, d_bounds])
        # Minimize negative soft-max (equivalent to maximizing soft-min)
        beta = 800.0
        d_max = np.max(all_d)
        shifted = d_max - all_d
        weights = np.exp(-beta * shifted)
        sum_w = np.sum(weights)
        soft_max = d_max + np.log(sum_w) / beta
        return -soft_max
    
    def objective_grad(params):
        c_p = params.reshape(N, 2)
        x = c_p[:, 0]
        y = c_p[:, 1]
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1.0)
        
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        d_bounds = np.concatenate([d_left, d_right, d_down, d_up])
        
        all_d = np.concatenate([d_pairs, d_bounds])
        beta = 800.0
        d_max = np.max(all_d)
        shifted = d_max - all_d
        weights = np.exp(-beta * shifted)
        sum_w = np.sum(weights)
        p = weights / sum_w
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        i_idx, j_idx = iu
        vec_x = x[i_idx] - x[j_idx]
        vec_y = y[i_idx] - y[j_idx]
        inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
        
        p_pairs = p[:len(d_pairs)]
        
        # For soft-max, gradient sign is flipped
        grad_x[i_idx] += p_pairs * vec_x * inv_d
        grad_x[j_idx] -= p_pairs * vec_x * inv_d
        grad_y[i_idx] += p_pairs * vec_y * inv_d
        grad_y[j_idx] -= p_pairs * vec_y * inv_d
        
        p_bounds = p[len(d_pairs):]
        p_left = p_bounds[:N]
        p_right = p_bounds[N:2*N]
        p_down = p_bounds[2*N:3*N]
        p_up = p_bounds[3*N:]
        
        grad_x += p_left - p_right
        grad_y += p_down - p_up
        
        total_grad = np.concatenate([grad_x, grad_y])
        return -total_grad  # Negate because we're minimizing -soft_max
    
    # Constraints: all pairwise distances >= 2*r_target, boundary distances >= r_target
    constraints = []
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    # Only add constraints for pairs that are close to the target
    # This keeps the constraint set manageable
    x = c[:, 0]
    y = c[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    d_pairs = d[iu]
    
    # Add constraints for pairs within 1.1 * 2*r_target (tighter)
    close_pairs = np.where(d_pairs < 1.1 * 2 * r_target)[0]
    for idx in close_pairs[:300]:  # Allow more constraints
        ii = i_idx[idx]
        jj = j_idx[idx]
        def pair_constraint(params, ii=ii, jj=jj):
            c_p = params.reshape(N, 2)
            return np.linalg.norm(c_p[ii] - c_p[jj]) - 2 * r_target
        constraints.append({'type': 'ineq', 'fun': pair_constraint})
    
    # Boundary constraints
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    close_bounds = np.where(np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up)) < 1.1 * r_target)[0]
    for i in close_bounds:
        def bound_constraint(params, i=i):
            c_p = params.reshape(N, 2)
            return np.min([0.5 - c_p[i, 0], 0.5 + c_p[i, 0], 0.5 - c_p[i, 1], 0.5 + c_p[i, 1]]) - r_target
        constraints.append({'type': 'ineq', 'fun': bound_constraint})
    
    try:
        res = minimize(
            objective,
            c.flatten(),
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c
>>>>>>> REPLACE

