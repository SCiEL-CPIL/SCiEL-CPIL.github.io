<<<<<<< SEARCH
def _penalty_objective_and_grad(x, w, h, penalty, smooth_eps=0.01):
    """
    Vectorized penalty objective function with analytical gradient.
    Uses log-sum-exp smoothing for overlap constraints.
    x: array of shape (3*n,) [x0, y0, r0, x1, y1, r1, ...]
    Returns (objective, gradient)
    """
    n = len(x) // 3
    xs = x[0::3].copy()
    ys = x[1::3].copy()
    rs = x[2::3].copy()
    
    # Objective: maximize sum of radii => minimize -sum(rs)
    obj = -np.sum(rs)
    grad = np.zeros_like(x)
    grad[2::3] = -1.0  # d(-sum(rs))/d(rs) = -1
    
    # 1. Boundary penalties (smooth max via softplus)
    # v1 = r - x (left), v2 = x + r - w (right), v3 = r - y (bottom), v4 = y + r - h (top)
    v_left = rs - xs
    v_right = xs + rs - w
    v_bottom = rs - ys
    v_top = ys + rs - h
    
    # Softplus: log(1 + exp(v/eps)) * eps ≈ max(0, v) for small eps
    def softplus(v, eps=smooth_eps):
        return eps * np.log1p(np.exp(np.clip(v / eps, -50, 50)))
    
    def softplus_grad(v, eps=smooth_eps):
        return 1.0 / (1.0 + np.exp(-np.clip(v / eps, -50, 50)))
    
    pen_boundary = softplus(v_left) + softplus(v_right) + softplus(v_bottom) + softplus(v_top)
    obj += penalty * np.sum(pen_boundary)
    
    # Gradients for boundary
    gl = penalty * softplus_grad(v_left)
    gr = penalty * softplus_grad(v_right)
    gb = penalty * softplus_grad(v_bottom)
    gt = penalty * softplus_grad(v_top)
    
    # d(v_left)/dxs = -1, d(v_left)/drs = 1
    grad[0::3] += -gl
    grad[2::3] += gl
    # d(v_right)/dxs = 1, d(v_right)/drs = 1
    grad[0::3] += gr
    grad[2::3] += gr
    # d(v_bottom)/dys = -1, d(v_bottom)/drs = 1
    grad[1::3] += -gb
    grad[2::3] += gb
    # d(v_top)/dys = 1, d(v_top)/drs = 1
    grad[1::3] += gt
    grad[2::3] += gt
    
    # 2. Overlap penalties with log-sum-exp smoothing
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_sq = dx**2 + dy**2
    dist = np.sqrt(dist_sq + 1e-12)
    
    min_dist = rs[:, None] + rs[None, :]
    
    # violation_ij = max(0, min_dist_ij - dist_ij)
    # Use smooth approximation: softplus(min_dist - dist)
    violations = min_dist - dist
    pen_overlap_vec = softplus(violations, eps=smooth_eps)
    
    # Only upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pen_overlap = np.sum(pen_overlap_vec[mask])
    obj += penalty * pen_overlap
    
    # Gradient of softplus(violation) w.r.t. x_i, y_i, r_i, x_j, y_j, r_j
    # d(softplus(v))/dv = sigmoid(v/eps)
    sp_grad = softplus_grad(violations, eps=smooth_eps)
    
    # d(violation)/dx_i = -dx_ij / dist_ij (since violation = ri+rj - dist, d(-dist)/dx_i = -dx_ij/dist)
    # Actually violation = min_dist - dist, so d(violation)/dx_i = -d(dist)/dx_i = -dx_ij/dist
    # d(violation)/dx_j = +dx_ij/dist
    # d(violation)/dr_i = 1, d(violation)/dr_j = 1
    
    inv_dist = 1.0 / dist
    dx_over_dist = dx * inv_dist
    dy_over_dist = dy * inv_dist
    
    # Gradient contribution for each pair (i,j) with i<j
    # For circle i: grad_xi += penalty * sp_grad[i,j] * (-dx_over_dist[i,j])
    #              grad_yi += penalty * sp_grad[i,j] * (-dy_over_dist[i,j])
    #              grad_ri += penalty * sp_grad[i,j] * 1
    # For circle j: grad_xj += penalty * sp_grad[i,j] * (+dx_over_dist[i,j])
    #              grad_yj += penalty * sp_grad[i,j] * (+dy_over_dist[i,j])
    #              grad_rj += penalty * sp_grad[i,j] * 1
    
    # Vectorized accumulation
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = np.zeros(n)
    
    # For upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            s = penalty * sp_grad[i, j]
            if s < 1e-10:
                continue
            grad_x[i] += -s * dx_over_dist[i, j]
            grad_y[i] += -s * dy_over_dist[i, j]
            grad_r[i] += s
            grad_x[j] += s * dx_over_dist[i, j]
            grad_y[j] += s * dy_over_dist[i, j]
            grad_r[j] += s
    
    grad[0::3] += grad_x
    grad[1::3] += grad_y
    grad[2::3] += grad_r
    
    return obj, grad

def _separate_circles(x, w, h, iterations=100):
    """
    Pairwise separation repair: push overlapping circles apart.
    x: flat array of shape (3*n,)
    """
    n = len(x) // 3
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = x[i*3], x[i*3+1], x[i*3+2]
                xj, yj, rj = x[j*3], x[j*3+1], x[j*3+2]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = ri + rj
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    # Push apart by half the overlap each
                    push_x = 0.5 * overlap * dx / dist
                    push_y = 0.5 * overlap * dy / dist
                    x[i*3] += push_x
                    x[i*3+1] += push_y
                    x[j*3] -= push_x
                    x[j*3+1] -= push_y
                    moved = True
                elif dist <= 1e-10:
                    # Random small push
                    x[i*3] += 0.01
                    x[j*3] -= 0.01
                    moved = True
        # Clamp to bounds
        for i in range(n):
            x[i*3] = np.clip(x[i*3], 0, w)
            x[i*3+1] = np.clip(x[i*3+1], 0, h)
            x[i*3+2] = np.clip(x[i*3+2], 0, min(w, h)/2)
        if not moved:
            break
    return x

def _optimize_penalty(x0, w, h, deadline, max_iter=300, penalty_start=5.0, penalty_growth=1.5):
    """
    Run L-BFGS-B with increasing penalty and analytical gradients.
    """
    x = x0.copy()
    penalty = penalty_start
    smooth_eps = 0.05
    
    for stage in range(8):
        if time.time() > deadline - 5.0:
            break
        
        # Reduce smoothing as penalty increases
        smooth_eps = max(0.005, 0.05 * (0.7 ** stage))
        
        def fun(x):
            obj, _ = _penalty_objective_and_grad(x, w, h, penalty, smooth_eps)
            return obj
        
        def jac(x):
            _, g = _penalty_objective_and_grad(x, w, h, penalty, smooth_eps)
            return g
            
        # Bounds
        bounds = []
        for _ in range(len(x0)//3):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0, min(w, h)/2))
            
        try:
            res = minimize(
                fun, x, jac=jac, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x = res.x
        except Exception:
            break
        
        # Apply separation repair
        x = _separate_circles(x, w, h, iterations=50)
        
        # Check validity
        circles = np.array([(x[i*3], x[i*3+1], x[i*3+2]) for i in range(len(x0)//3)])
        if _validate(circles, w, h, tol=1e-4):
            break
        else:
            penalty *= penalty_growth
            
    return x

def _slsqp_polish(circles, w, h, deadline):
    """
    High-precision polish with SLSQP.
    """
    if time.time() > deadline - 3.0:
        return circles
        
    n = len(circles)
    x0 = np.array([v for c in circles for v in c])
    
    def objective(x):
        return -np.sum(x[2::3])
        
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-8})
        
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })
            
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-8, min(w, h)/2))
        
    try:
        res = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-14}
        )
        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
            if _validate(new_circles, w, h, tol=1e-8):
                return new_circles
        # Even if not successful, check if the result is valid and better
        new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
        if _validate(new_circles, w, h, tol=1e-8):
            if _total_radius(new_circles) > _total_radius(circles):
                return new_circles
    except Exception:
        pass
        
    return circles

def _solve_aspect(w, h, t0, deadline, x_init=None):
    # Use provided initialization or default to hexagonal grid
    if x_init is None:
        x_init = _hexagonal_init(w, h, N)
    
    # Flatten for optimizer
    x_flat = x_init.flatten()
    
    # Optimize with penalty method
    x_opt = _optimize_penalty(x_flat, w, h, deadline)
    
    # Reshape
    circles = np.array([(x_opt[i*3], x_opt[i*3+1], x_opt[i*3+2]) for i in range(N)])
    
    # Polish with SLSQP if time permits
    if time.time() < deadline - 5.0:
        circles = _slsqp_polish(circles, w, h, deadline)
        
    return circles
=======
def _refine_packing(circles, w, h, deadline):
    """
    Robust local search to maximize sum of radii while maintaining validity.
    """
    n = len(circles)
    if n == 0:
        return circles
    
    # Work with a copy
    c = circles.copy()
    
    # Initial validation and repair
    if not _validate(c, w, h, tol=1e-5):
        # If invalid, shrink radii slightly to make valid
        for i in range(n):
            c[i, 2] *= 0.9
        if not _validate(c, w, h, tol=1e-5):
            # Last resort: grid fallback
            side = int(np.ceil(np.sqrt(n)))
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) >= n:
                        break
                    cx = (i + 0.5) * w / side
                    cy = (j + 0.5) * h / side
                    r = 0.5 * min(w, h) / side * 0.9
                    cells.append([cx, cy, r])
            return np.array(cells[:n])
    
    # Local search loop
    step_size = 0.01
    no_improve_count = 0
    
    while time.time() < deadline - 2.0:
        improved = False
        
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            
            xi, yi, ri = c[i, 0], c[i, 1], c[i, 2]
            
            # Calculate maximum possible radius for circle i given current positions of others
            # and walls.
            r_max = min(w - xi, xi, h - yi, yi)
            
            for j in range(n):
                if i == j:
                    continue
                xj, yj, rj = c[j, 0], c[j, 1], c[j, 2]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                # dist >= ri + rj => ri <= dist - rj
                if dist > rj:
                    r_max = min(r_max, dist - rj)
            
            # If we can increase radius, do so
            if r_max > ri + 1e-6:
                # Increase by a fraction of the gap to be safe
                new_r = ri + 0.5 * (r_max - ri)
                # Ensure it doesn't violate bounds
                new_r = min(new_r, w - xi, xi, h - yi, yi)
                # Ensure it doesn't overlap with others
                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    xj, yj, rj = c[j, 0], c[j, 1], c[j, 2]
                    dx = xi - xj
                    dy = yi - yj
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_r + rj - 1e-7:
                        valid = False
                        break
                if valid:
                    c[i, 2] = new_r
                    improved = True
            else:
                # Try moving the circle to find a spot where it can be larger
                # Direction: towards the center of the "free space"
                # Simple heuristic: move towards the centroid of the rectangle
                # or random small perturbations
                # We will try small random perturbations in x and y
                for _ in range(5):
                    dx_move = np.random.uniform(-step_size, step_size)
                    dy_move = np.random.uniform(-step_size, step_size)
                    new_x = xi + dx_move
                    new_y = yi + dy_move
                    
                    # Check bounds
                    if new_x < ri or new_x > w - ri or new_y < ri or new_y > h - ri:
                        continue
                    
                    # Check overlaps
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        xj, yj, rj = c[j, 0], c[j, 1], c[j, 2]
                        dx = new_x - xj
                        dy = new_y - yj
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < ri + rj - 1e-7:
                            valid = False
                            break
                    
                    if valid:
                        # Check if this move allows for a larger radius
                        r_max_new = min(w - new_x, new_x, h - new_y, new_y)
                        for j in range(n):
                            if i == j:
                                continue
                            xj, yj, rj = c[j, 0], c[j, 1], c[j, 2]
                            dx = new_x - xj
                            dy = new_y - yj
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist > rj:
                                r_max_new = min(r_max_new, dist - rj)
                        
                        if r_max_new > ri + 1e-6:
                            c[i, 0] = new_x
                            c[i, 1] = new_y
                            c[i, 2] = ri + 0.5 * (r_max_new - ri)
                            improved = True
                            break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 10:
                break
            # Reduce step size if stuck
            step_size *= 0.5
            if step_size < 1e-5:
                break
        else:
            no_improve_count = 0
    
    return c

def _solve_aspect(w, h, t0, deadline, x_init=None):
    # Use provided initialization or default to hexagonal grid
    if x_init is None:
        x_init = _hexagonal_init(w, h, N)
    
    # Refine the packing
    circles = _refine_packing(x_init, w, h, deadline)
    
    return circles
>>>>>>> REPLACE

