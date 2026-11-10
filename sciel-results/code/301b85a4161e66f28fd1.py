<<<<<<< SEARCH
def _optimize_from_init(pts_init, time_budget, rng):
    """Optimize starting from pts_init using L-BFGS-B in tangent space"""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # First do a few steps of projected gradient descent to get close
    n = pts_init.shape[0]
    pts = pts_init.copy()
    
    # Initial gradient descent phase
    for step in range(200):
        if time.time() > deadline - 5:
            break
        grad = _log_of_exp_map_grad(pts, np.zeros((n, 3)))
        # Simple gradient descent
        lr = 0.01
        new_pts = pts - lr * grad
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / np.maximum(norms, 1e-15)
        pts = new_pts
    
    # Now use L-BFGS-B in tangent space
    best_energy = _log_energy_vec(pts)
    best_pts = pts.copy()
    
    # Flatten for L-BFGS
    x0 = np.zeros(3 * N)
    
    def fun_and_grad(x_flat):
        deltas = x_flat.reshape(N, 3)
        # Project deltas to be tangent to pts
        dots = np.sum(deltas * pts, axis=1)
        deltas_tangent = deltas - dots[:, np.newaxis] * pts
        
        energy = _log_of_exp_map(pts, deltas_tangent)
        grad = _log_of_exp_map_grad(pts, deltas_tangent)
        return energy, grad.flatten()
    
    try:
        result = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        if time.time() < deadline - 3:
            deltas_opt = result.x.reshape(N, 3)
            dots = np.sum(deltas_opt * pts, axis=1)
            deltas_opt = deltas_opt - dots[:, np.newaxis] * pts
            new_pts = _exp_map(pts, deltas_opt)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.maximum(norms, 1e-15)
            new_energy = _log_energy_vec(new_pts)
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
    except Exception:
        pass
    
    return best_pts, best_energy
=======
def _optimize_from_init(pts_init, time_budget, rng):
    """Optimize starting from pts_init using L-BFGS-B in tangent space"""
    start_time = time.time()
    deadline = start_time + time_budget
    
    n = pts_init.shape[0]
    pts = pts_init.copy()
    
    # Quick gradient descent with adaptive step to get into a good basin
    for step in range(100):
        if time.time() > deadline - 10:
            break
        grad = _log_of_exp_map_grad(pts, np.zeros((n, 3)))
        # Adaptive step size based on gradient magnitude
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        lr = min(0.05, 0.01 / max(grad_norm, 1e-8))
        new_pts = pts - lr * grad
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / np.maximum(norms, 1e-15)
        
        # Check if this improves energy
        old_e = _log_energy_vec(pts)
        new_e = _log_energy_vec(new_pts)
        if new_e < old_e:
            pts = new_pts
        else:
            # Try smaller step
            new_pts2 = pts - lr * 0.1 * grad
            norms2 = np.linalg.norm(new_pts2, axis=1, keepdims=True)
            new_pts2 = new_pts2 / np.maximum(norms2, 1e-15)
            new_e2 = _log_energy_vec(new_pts2)
            if new_e2 < old_e:
                pts = new_pts2
            else:
                break
    
    # Now use L-BFGS-B in tangent space
    best_energy = _log_energy_vec(pts)
    best_pts = pts.copy()
    
    # Flatten for L-BFGS
    x0 = np.zeros(3 * N)
    
    def fun_and_grad(x_flat):
        deltas = x_flat.reshape(N, 3)
        # Project deltas to be tangent to pts
        dots = np.sum(deltas * pts, axis=1)
        deltas_tangent = deltas - dots[:, np.newaxis] * pts
        
        energy = _log_of_exp_map(pts, deltas_tangent)
        grad = _log_of_exp_map_grad(pts, deltas_tangent)
        return energy, grad.flatten()
    
    try:
        result = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000}
        )
        if time.time() < deadline - 3:
            deltas_opt = result.x.reshape(N, 3)
            dots = np.sum(deltas_opt * pts, axis=1)
            deltas_opt = deltas_opt - dots[:, np.newaxis] * pts
            new_pts = _exp_map(pts, deltas_opt)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.maximum(norms, 1e-15)
            new_energy = _log_energy_vec(new_pts)
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
    except Exception:
        pass
    
    return best_pts, best_energy
>>>>>>> REPLACE

