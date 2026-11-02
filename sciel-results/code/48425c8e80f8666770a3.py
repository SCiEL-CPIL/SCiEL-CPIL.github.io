# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _objective_slsqp(params):
    """Objective: minimize -sum(r)."""
    r = params[2::3]
    return -np.sum(r)

def _objective_slsqp_grad(params):
    """Gradient of objective."""
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    return grad

def _get_constraints_and_grads():
    """
    Generate constraint functions and gradients for SLSQP.
    Constraints are inequalities: g(params) >= 0.
    """
    constraints = []
    
    # 1. Boundary constraints
    # x_i - r_i >= 0
    # x_i + r_i <= 1  => 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # y_i + r_i <= 1  => 1 - y_i - r_i >= 0
    # r_i >= 0
    
    for i in range(N):
        # x_i - r_i >= 0
        def c_xmin(params, i=i):
            return params[3*i] - params[3*i+2]
        def g_xmin(params, i=i):
            g = np.zeros_like(params)
            g[3*i] = 1.0
            g[3*i+2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_xmin, 'jac': g_xmin})
        
        # 1 - x_i - r_i >= 0
        def c_xmax(params, i=i):
            return 1.0 - params[3*i] - params[3*i+2]
        def g_xmax(params, i=i):
            g = np.zeros_like(params)
            g[3*i] = -1.0
            g[3*i+2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_xmax, 'jac': g_xmax})
        
        # y_i - r_i >= 0
        def c_ymin(params, i=i):
            return params[3*i+1] - params[3*i+2]
        def g_ymin(params, i=i):
            g = np.zeros_like(params)
            g[3*i+1] = 1.0
            g[3*i+2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_ymin, 'jac': g_ymin})
        
        # 1 - y_i - r_i >= 0
        def c_ymax(params, i=i):
            return 1.0 - params[3*i+1] - params[3*i+2]
        def g_ymax(params, i=i):
            g = np.zeros_like(params)
            g[3*i+1] = -1.0
            g[3*i+2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_ymax, 'jac': g_ymax})
        
        # r_i >= 0
        def c_rmin(params, i=i):
            return params[3*i+2]
        def g_rmin(params, i=i):
            g = np.zeros_like(params)
            g[3*i+2] = 1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_rmin, 'jac': g_rmin})
        
    # 2. Non-overlap constraints: dist(i,j) - r_i - r_j >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def c_overlap(params, i=i, j=j):
                x_i, y_i, r_i = params[3*i], params[3*i+1], params[3*i+2]
                x_j, y_j, r_j = params[3*j], params[3*j+1], params[3*j+2]
                dx = x_i - x_j
                dy = y_i - y_j
                d = np.sqrt(dx*dx + dy*dy)
                return d - r_i - r_j
            def g_overlap(params, i=i, j=j):
                g = np.zeros_like(params)
                x_i, y_i, r_i = params[3*i], params[3*i+1], params[3*i+2]
                x_j, y_j, r_j = params[3*j], params[3*j+1], params[3*j+2]
                dx = x_i - x_j
                dy = y_i - y_j
                d_sq = dx*dx + dy*dy
                if d_sq < 1e-12:
                    d = 1e-6
                    dx, dy = 1e-6, 0.0
                else:
                    d = np.sqrt(d_sq)
                
                g[3*i] = dx / d
                g[3*i+1] = dy / d
                g[3*i+2] = -1.0
                
                g[3*j] = -dx / d
                g[3*j+1] = -dy / d
                g[3*j+2] = -1.0
                return g
            constraints.append({'type': 'ineq', 'fun': c_overlap, 'jac': g_overlap})
            
    return constraints

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _optimize_slsqp(xyz, time_budget=10.0):
    """
    Optimize using SLSQP with analytical gradients and full constraints.
    """
    start_time = time.time()
    params = xyz.flatten()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
        
    constraints = _get_constraints_and_grads()
    
    try:
        result = minimize(_objective_slsqp, params, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          jac=_objective_slsqp_grad,
                          options={'maxiter': 200, 'ftol': 1e-9})
        
        if result.success or np.sum(result.x[2::3]) > np.sum(params[2::3]):
            return result.x.reshape(N, 3)
    except Exception:
        pass
        
    return xyz

def _fix_config(xyz):
    """Try to fix an invalid configuration by reducing radii and adjusting positions."""
    xyz_fixed = xyz.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        xyz_fixed[:, 2] *= 0.95
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Random configurations
    np.random.seed(42)
    for trial in range(4):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        time_budget = min(15.0, remaining - 2.0)
        if time_budget < 2.0:
            break
        
        xyz_opt = _optimize_slsqp(xyz0, time_budget=time_budget)
        
        # Fix if needed
        if not _validate(xyz_opt):
            xyz_opt = _fix_config(xyz_opt)
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
            
    # Final validation and fix
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
        
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
        
    return best_xyz

# EVOLVE-BLOCK-END
