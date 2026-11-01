<<<<<<< SEARCH
def _local_optimize(pts, max_iter=500, tol=1e-12):
    """
    Local optimization using gradient ascent on the smooth surrogate.
    We maximize f(x) = min(1 - |x_i|, min_{i<j} |x_i - x_j|/2).
    To make it smooth, we can use a log-sum-exp approximation for the min,
    or simply use a penalty method / direct gradient of the active set.
    
    A simpler and robust approach for circle packing is to use a force-directed
    relaxation or a specific optimization library. Since SciPy is available,
    let's use `minimize` with a negative objective (maximization) and a
    smooth approximation of the min function:
    min(a, b) ≈ -log(exp(-a/tau) + exp(-b/tau)) * tau, as tau -> 0.
    
    However, implementing a custom gradient for the min is tricky.
    Alternative: Use `scipy.optimize.differential_evolution`? No, too slow for 600D.
    
    Let's use a simple iterative repulsion method (like molecular dynamics)
    or a gradient-based method where we compute the gradient of the objective
    using the active constraints.
    
    Actually, a very effective method for this is to use `scipy.optimize.minimize`
    with the objective being the negative of the soft-min, or simply maximizing
    the minimum distance.
    
    Let's try a gradient ascent approach where at each step, we compute the
    current radius r, identify the "bottleneck" constraints, and move points
    in the direction that increases r.
    
    For a more robust and standard approach, let's use `scipy.optimize.minimize`
    with the method 'L-BFGS-B' or 'Powell' on the negative of the radius.
    But the radius function is non-smooth.
    
    Let's use a soft-min approach:
    r_soft = -tau * log( sum(exp(-c_i/tau)) )
    where c_i are the constraint values (1-|x_i| and |x_i-x_j|/2).
    This is smooth and differentiable.
    """
    tau = 0.01  # Temperature for soft-min
    best_pts = pts.copy()
    best_r = _score(pts)
    
    def objective(x):
        pts_flat = x.reshape(N, 2)
        # Compute constraint values
        rad = np.hypot(pts_flat[:, 0], pts_flat[:, 1])
        c_cont = 1.0 - rad  # (N,)
        
        # Pairwise distances
        diff = pts_flat[:, None, :] - pts_flat[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        c_pair = dist / 2.0  # (N, N)
        
        # Soft-min of all constraints
        # To avoid overflow, use the log-sum-exp trick
        all_c = np.concatenate([c_cont, c_pair.flatten()])
        # Filter out non-finite or very small values if necessary, but soft-min handles it.
        m = np.min(all_c)
        if m < -1000:  # Safety check
            return -m  # Return a large negative value (bad score)
        
        # r_soft = m - tau * log( sum(exp((c_i - m)/tau)) )
        # We want to maximize r_soft, so minimize -r_soft
        sum_exp = np.sum(np.exp((all_c - m) / tau))
        r_soft = m - tau * np.log(sum_exp)
        return -r_soft
    
    # Use L-BFGS-B with bounds? No explicit bounds needed if we penalize going outside,
    # but the soft-min will naturally discourage it.
    # However, L-BFGS-B requires bounds. Let's use 'BFGS' or 'Powell'.
    # 'Powell' is derivative-free and robust for non-smooth functions, but slow.
    # 'L-BFGS-B' is faster but needs smoothness. Our soft-min is smooth.
    # Let's provide bounds to keep points in the disk.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        res = minimize(objective, pts.flatten(), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        if res.success and _score(res.x.reshape(N, 2)) > best_r:
            best_pts = res.x.reshape(N, 2)
            best_r = _score(best_pts)
    except Exception:
        pass
    
    return best_pts, best_r
=======
def _local_optimize(pts, max_iter=1000, deadline=None):
    """
    Fast local optimization using a repulsion-based relaxation (force-directed).
    This is much faster than L-BFGS for this problem size.
    """
    if deadline is None:
        deadline = time.time() + 10.0  # Default 10s limit for this function
    
    best_pts = pts.copy()
    best_r = _score(pts)
    current_pts = pts.copy()
    
    # Learning rate / step size
    lr = 0.01
    # Decay rate for learning rate
    decay = 0.99
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute forces
        forces = np.zeros_like(current_pts)
        
        # 1. Repulsion between circles
        # We want to maximize the minimum distance.
        # A simple heuristic: push points apart if they are too close.
        # Or, compute gradient of a soft-min.
        # Let's use a simple repulsive force for pairs closer than some threshold.
        # However, to be robust, we can use the gradient of the soft-min.
        
        # Recompute distances
        diff = current_pts[:, None, :] - current_pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Soft-min gradient
        tau = 0.01
        # To avoid memory issues and speed up, we can approximate the gradient.
        # The gradient of the soft-min w.r.t. point k is a weighted sum of unit vectors to its neighbors.
        # weights w_ij = exp(-d_ij / tau) / sum_j exp(-d_ij / tau)
        
        # For efficiency, we can just use a simple repulsion for all pairs, scaled by distance.
        # F_rep = sum_j (1/d_ij^2) * (p_i - p_j)  for d_ij < 2r_est
        # But we don't know r_est exactly.
        
        # Let's stick to the soft-min gradient but compute it efficiently.
        # Actually, for N=300, even O(N^2) per step is ~90k ops. 1000 steps = 90M ops.
        # Python loops are slow. Vectorized NumPy is fast.
        
        # Calculate weights for soft-min
        # We need the gradient of r_soft = -tau * log(sum(exp(-c/tau)))
        # c_i are distances.
        # d(r_soft)/d(p_k) = (1/tau) * sum_j w_j * (p_j - p_k) / d_jk * (1/tau) ? 
        # Actually, d(c_ij)/d(p_k) = (p_k - p_j) / d_ij for i=k, j!=k.
        
        # Let's use a simpler, proven method: 
        # Move each point in the direction of the average of its neighbors, 
        # weighted by inverse distance, but OUTWARD to increase distance?
        # No, we want to increase the MINIMUM distance.
        # So we should move points away from their closest neighbors.
        
        # Simplest effective heuristic:
        # For each point, find its closest neighbor. Move away from it.
        # This is O(N^2) to find closest, then O(N) to update.
        
        # Vectorized closest neighbor
        min_dist, min_idx = np.min(dist, axis=1), np.argmin(dist, axis=1)
        
        # Direction away from closest neighbor
        dirs = current_pts - current_pts[min_idx]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dirs = dirs / norms
        
        # Force from closest neighbor
        f_closest = dirs
        
        # Also add a force to keep points inside the disk
        rad = np.linalg.norm(current_pts, axis=1, keepdims=True)
        f_container = -current_pts / rad * 0.5 # Weak attraction to center? No, repulsion from boundary.
        # If a point is close to the boundary, push it in.
        # Boundary is at r=1. If r > 1 - r_est, push in.
        # Let's just add a small attraction to the center to prevent drift, 
        # but the main constraint is the pairwise distance.
        
        # Update
        current_pts += lr * (f_closest)
        
        # Project back to disk if necessary (hard constraint)
        rad_new = np.linalg.norm(current_pts, axis=1, keepdims=True)
        mask = rad_new > 0.99
        current_pts[mask] = current_pts[mask] * (0.99 / rad_new[mask])
        
        # Evaluate
        r_curr = _score(current_pts)
        if r_curr > best_r:
            best_r = r_curr
            best_pts = current_pts.copy()
        else:
            # If score dropped, revert to best and reduce step size
            current_pts = best_pts.copy()
            lr *= 0.9
        
        # Decay learning rate
        lr *= decay
    
    return best_pts, best_r
>>>>>>> REPLACE

