import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _normalize_batch(pts):
    """Normalize batch of vectors."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_hull(config):
    """
    Compute covering radius in degrees using convex hull facets.
    Returns (radius_deg, circumcenters_array, facet_indices_array)
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([]), np.array([], dtype=int)
    
    simplices = hull.simplices
    n_facets = len(simplices)
    
    # Vectorized circumcenter computation
    p0 = config[simplices[:, 0]]  # (F, 3)
    p1 = config[simplices[:, 1]]
    p2 = config[simplices[:, 2]]
    
    v1 = p1 - p0  # (F, 3)
    v2 = p2 - p0  # (F, 3)
    
    # Cross product for each facet
    c = np.cross(v1, v2)  # (F, 3)
    c_norms = np.linalg.norm(c, axis=1)
    
    # Filter out degenerate facets
    valid_mask = c_norms > 1e-12
    if not np.any(valid_mask):
        return 0.0, np.array([]), np.array([], dtype=int)
    
    c = c[valid_mask]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    valid_facets = simplices[valid_mask]
    
    # Check if circumcenter is equidistant from all 3 vertices
    dots0 = np.einsum('ij,ij->i', c, p0[valid_mask])
    dots1 = np.einsum('ij,ij->i', c, p1[valid_mask])
    dots2 = np.einsum('ij,ij->i', c, p2[valid_mask])
    
    dots = np.column_stack([dots0, dots1, dots2])  # (F, 3)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (F, 3)
    
    # Valid circumcenters: all three angles should be approximately equal
    angle_spread = np.max(angles, axis=1) - np.min(angles, axis=1)
    valid_circ = angle_spread < 1e-6
    
    if not np.any(valid_circ):
        grid = _fib_sample(10000)
        dots_grid = grid @ config.T
        min_dots = np.min(dots_grid, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([]), np.array([], dtype=int)
    
    c = c[valid_circ]
    valid_facets = valid_facets[valid_circ]
    
    # For each valid circumcenter, find nearest config point
    dots_all = c @ config.T  # (F, N)
    dots_all = np.clip(dots_all, -1.0, 1.0)
    max_dots = np.max(dots_all, axis=1)  # (F,)
    min_angles = np.arccos(max_dots)  # (F,)
    
    max_angle = np.max(min_angles)
    radius_deg = np.degrees(max_angle)
    
    return radius_deg, c, valid_facets

def _soft_max_surrogate(config, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp on facet circumradii.
    Returns (value, gradient)
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        # Fallback to grid-based if hull fails
        grid = _fib_sample(5000)
        dots = grid @ config.T # (G, N)
        min_dots = np.min(dots, axis=1)
        angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
        # Soft max of angles
        beta_val = 50.0
        m = np.max(angles)
        exp_angles = np.exp(beta_val * (angles - m))
        val = m + np.log(np.sum(exp_angles)) / beta_val
        
        # Gradient wrt config
        # angles = arccos(min_j dot(grid_i, config_j))
        # d angle / d config_j = - dot(grid_i, config_j) / sqrt(1 - dot^2) * grid_i (if j is the min)
        # This is complex. Let's just return a large penalty and zero grad if hull fails.
        return 100.0, np.zeros_like(config)

    simplices = hull.simplices
    p0 = config[simplices[:, 0]]
    p1 = config[simplices[:, 1]]
    p2 = config[simplices[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    c = np.cross(v1, v2)
    c_norms = np.linalg.norm(c, axis=1)
    valid_mask = c_norms > 1e-12
    
    if not np.any(valid_mask):
        return 0.0, np.zeros_like(config)
        
    c = c[valid_mask]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    valid_facets = simplices[valid_mask]
    
    # Compute circumradius (angle) for each valid facet
    # Angle = arccos(dot(c, p0))
    dots0 = np.einsum('ij,ij->i', c, p0[valid_mask])
    dots1 = np.einsum('ij,ij->i', c, p1[valid_mask])
    dots2 = np.einsum('ij,ij->i', c, p2[valid_mask])
    
    dots = np.column_stack([dots0, dots1, dots2])
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    # Check validity: angles should be equal
    angle_spread = np.max(angles, axis=1) - np.min(angles, axis=1)
    valid_circ = angle_spread < 1e-6
    
    if not np.any(valid_circ):
        return 100.0, np.zeros_like(config)
        
    c = c[valid_circ]
    valid_facets = valid_facets[valid_circ]
    angles = angles[valid_circ, 0] # Use first angle, they are approx equal
    
    # Soft max of angles
    beta_val = 50.0
    m = np.max(angles)
    exp_angles = np.exp(beta_val * (angles - m))
    sum_exp = np.sum(exp_angles)
    val = m + np.log(sum_exp) / beta_val
    
    # Gradient
    # d val / d angle_k = exp(beta * (angle_k - m)) / sum_exp
    weights = exp_angles / sum_exp
    
    # d angle_k / d config
    # angle_k = arccos(dot(c_k, p_{k,0}))
    # c_k depends on p_{k,0}, p_{k,1}, p_{k,2}
    # This is complicated. Let's use a simpler surrogate:
    # Minimize the soft max of the squared distances from a grid to the points.
    # This is easier to differentiate.
    
    # Let's restart the surrogate definition to be grid-based for simplicity and robustness of gradient.
    # The grid-based surrogate is a lower bound on the true covering radius (if grid is fine enough)
    # and is smooth.
    
    grid = _fib_sample(2000) # Moderate size for speed
    dots = grid @ config.T # (G, N)
    min_dots = np.min(dots, axis=1)
    angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
    
    beta_val = 50.0
    m = np.max(angles)
    exp_angles = np.exp(beta_val * (angles - m))
    sum_exp = np.sum(exp_angles)
    val = m + np.log(sum_exp) / beta_val
    
    # Gradient of val wrt config
    # val = m + (1/beta) * log(sum_j exp(beta * (angle_j - m)))
    # d val / d angle_j = exp(beta * (angle_j - m)) / sum_exp = w_j
    w = exp_angles / sum_exp # (G,)
    
    # d angle_j / d config_i
    # angle_j = arccos( min_k dot(grid_j, config_k) )
    # Let k_j be the argmin index for grid point j.
    # dot_j = dot(grid_j, config_{k_j})
    # angle_j = arccos(dot_j)
    # d angle_j / d config_{k_j} = -1/sqrt(1-dot_j^2) * grid_j
    
    # We need to accumulate gradients for each config point.
    grad = np.zeros_like(config)
    
    # Find argmin for each grid point
    argmins = np.argmin(dots, axis=1)
    
    # For each grid point j, the gradient contribution to config[argmins[j]] is:
    # w_j * d angle_j / d config[argmins[j]]
    # = w_j * (-1/sqrt(1 - dots[j, argmins[j]]^2)) * grid[j]
    
    # Vectorized:
    # dots_min = dots[np.arange(G), argmins]
    dots_min = dots[np.arange(len(grid)), argmins]
    dots_min_clipped = np.clip(dots_min, -1.0 + 1e-9, 1.0 - 1e-9)
    denom = np.sqrt(1.0 - dots_min_clipped**2)
    
    # grad_contrib[j] = -w[j] / denom[j] * grid[j]
    # We need to sum these contributions for each config point i.
    # grad[i] = sum_{j: argmins[j]==i} grad_contrib[j]
    
    # Use np.add.at for efficient accumulation
    grad_contrib = (-w / denom)[:, np.newaxis] * grid # (G, 3)
    
    np.add.at(grad, argmins, grad_contrib)
    
    # Project gradient to tangent space
    # grad[i] = grad[i] - dot(grad[i], config[i]) * config[i]
    grad_proj = np.einsum('ij,ij->i', grad, config)
    grad = grad - grad_proj[:, np.newaxis] * config
    
    return val, grad

def _optimize_lgbfgs(config, deadline):
    """
    Optimize using L-BFGS-B with a smooth surrogate.
    """
    config = config.copy()
    x0 = config.flatten()
    
    def fun_grad(x):
        cfg = x.reshape(N, 3)
        cfg = _normalize_batch(cfg)
        val, grad = _soft_max_surrogate(cfg)
        return val, grad.flatten()
    
    best_x = x0
    best_val, _ = fun_grad(x0)
    
    # Run L-BFGS
    # We might need to run it multiple times or with different settings
    # Since the landscape is non-convex, we'll run it for a fixed number of iterations
    # or until time limit.
    
    while time.time() < deadline - 10:
        res = minimize(fun_grad, best_x, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x
        else:
            # If it didn't improve, break or try a different start?
            # For now, break if no improvement.
            break
            
    best_cfg = _normalize_batch(best_x.reshape(N, 3))
    return best_cfg

def _local_optimize(config, deadline, min_time_remaining=10.0):
    """
    Local optimization using circumcenter-based critical points.
    """
    config = config.copy()
    current_score, _, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.04
    stagnation = 0
    max_stagnation = 30
    
    while time.time() < deadline - min_time_remaining:
        score, circumcenters, facets = _covering_radius_hull(config)
        
        if len(circumcenters) == 0:
            break
        
        current_score = score
        
        # Find the worst circumcenter
        dots_all = circumcenters @ config.T
        dots_all = np.clip(dots_all, -1.0, 1.0)
        max_dots = np.max(dots_all, axis=1)
        min_angles = np.arccos(max_dots)
        
        worst_idx = np.argmax(min_angles)
        worst_point = circumcenters[worst_idx]
        worst_angle = min_angles[worst_idx]
        
        # Find candidate points to move: the 3 nearest config points to worst_point
        dots_to_worst = np.clip(worst_point @ config.T, -1.0, 1.0)
        sorted_idx = np.argsort(-dots_to_worst)
        
        improved = False
        
        for trial in range(3):
            if trial >= len(sorted_idx):
                break
            idx = sorted_idx[trial]
            
            # Compute tangent direction toward worst_point
            v = worst_point - config[idx]
            v_proj = v - np.dot(v, config[idx]) * config[idx]
            v_norm = np.linalg.norm(v_proj)
            
            if v_norm < 1e-10:
                continue
            
            v_dir = v_proj / v_norm
            
            # Try a few step sizes
            for ss in [step_size, step_size * 0.5, step_size * 2.0]:
                new_pt = config[idx] + ss * v_dir
                new_pt = _normalize(new_pt)
                
                old_pt = config[idx].copy()
                config[idx] = new_pt
                
                new_score, _, _ = _covering_radius_hull(config)
                
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    stagnation = 0
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                    if new_score < current_score - 0.01:
                        step_size = min(step_size * 1.2, 0.1)
                    break
                else:
                    config[idx] = old_pt
            
            if improved:
                break
        
        if not improved:
            stagnation += 1
            step_size *= 0.85
            if step_size < 0.001 or stagnation > max_stagnation:
                break
    
    return best_config

def _polish(config, deadline, min_time_remaining=5.0):
    """
    High-precision polish pass with very small steps.
    """
    config = config.copy()
    current_score, _, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.005
    stagnation = 0
    
    while time.time() < deadline - min_time_remaining:
        score, circumcenters, facets = _covering_radius_hull(config)
        
        if len(circumcenters) == 0:
            break
        
        current_score = score
        
        dots_all = circumcenters @ config.T
        dots_all = np.clip(dots_all, -1.0, 1.0)
        max_dots = np.max(dots_all, axis=1)
        min_angles = np.arccos(max_dots)
        
        # Find all circumcenters with angles close to the maximum
        threshold = np.max(min_angles) - 0.001
        worst_mask = min_angles > threshold
        worst_indices = np.where(worst_mask)[0]
        
        if len(worst_indices) == 0:
            break
        
        improved = False
        
        for wi in worst_indices[:5]:
            worst_point = circumcenters[wi]
            
            dots_to_worst = np.clip(worst_point @ config.T, -1.0, 1.0)
            sorted_idx = np.argsort(-dots_to_worst)
            
            for trial in range(2):
                if trial >= len(sorted_idx):
                    break
                idx = sorted_idx[trial]
                
                v = worst_point - config[idx]
                v_proj = v - np.dot(v, config[idx]) * config[idx]
                v_norm = np.linalg.norm(v_proj)
                
                if v_norm < 1e-10:
                    continue
                
                v_dir = v_proj / v_norm
                
                new_pt = config[idx] + step_size * v_dir
                new_pt = _normalize(new_pt)
                
                old_pt = config[idx].copy()
                config[idx] = new_pt
                
                new_score, _, _ = _covering_radius_hull(config)
                
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    stagnation = 0
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                    break
                else:
                    config[idx] = old_pt
            
            if improved:
                break
        
        if not improved:
            stagnation += 1
            step_size *= 0.9
            if step_size < 0.0005 or stagnation > 50:
                break
    
    return best_config

def _optimize(config, deadline):
    """
    Full optimization pipeline: Lloyd's + L-BFGS + local optimization + polish.
    """
    config = config.copy()
    
    # Phase 1: Lloyd's algorithm for good initial configuration
    M = 10000
    grid = _fib_sample(M)
    
    for iteration in range(150):
        if time.time() > deadline - 50:
            break
        dots = grid @ config.T
        max_idx = np.argmax(dots, axis=1)
        
        new_config = np.zeros_like(config)
        for i in range(N):
            mask = (max_idx == i)
            count = np.sum(mask)
            if count > 0:
                center = np.mean(grid[mask], axis=0)
                new_config[i] = _normalize(center)
            else:
                new_config[i] = _normalize(np.random.randn(3))
        
        if np.allclose(config, new_config, atol=1e-8):
            break
        config = new_config
        
    # Phase 2: L-BFGS Optimization
    if time.time() < deadline - 30:
        config = _optimize_lgbfgs(config, deadline)
        
    # Phase 3: Local optimization
    if time.time() < deadline - 15:
        config = _local_optimize(config, deadline, min_time_remaining=10.0)
        
    # Phase 4: Polish
    if time.time() < deadline - 5:
        config = _polish(config, deadline, min_time_remaining=3.0)
        
    return config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with Fibonacci spiral
    config = _fib_sample(N)
    
    # Run optimization
    config = _optimize(config, deadline)
    
    # Final normalization and validation
    config = _normalize_batch(config)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
                
    return config
