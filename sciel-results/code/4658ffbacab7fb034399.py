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
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    max_angle = 0.0
    for simplex in hull.simplices:
        pts = config[simplex]
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            continue
        c = c / c_norm
        dots = np.clip(pts @ c, -1.0, 1.0)
        angles = np.arccos(dots)
        if np.max(angles) - np.min(angles) > 1e-6:
            max_in_facet = np.max(angles)
        else:
            max_in_facet = angles[0]
        dots_all = np.clip(config @ c, -1.0, 1.0)
        min_angle_to_c = np.arccos(np.max(dots_all))
        if min_angle_to_c > max_angle:
            max_angle = min_angle_to_c
    return np.degrees(max_angle)

def _soft_max_covering_radius(config, beta):
    """
    Compute a smooth approximation of the covering radius using log-sum-exp.
    beta controls the sharpness (higher = closer to true max).
    Returns (value, gradient_wrt_points)
    """
    # Use a dense grid for smooth approximation
    M = 10000
    grid = _fib_sample(M)
    
    # dots[i, j] = grid[i] . config[j]
    dots = grid @ config.T  # (M, N)
    
    # For each grid point, find max dot product (closest config point)
    # max_dots[i] = max_j dots[i, j]
    max_dots = np.max(dots, axis=1)
    
    # angles[i] = arccos(max_dots[i])
    angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
    
    # Soft max: log-sum-exp of exp(beta * angles)
    # This approximates max(angles) as beta -> inf
    beta_angles = beta * angles
    max_beta = np.max(beta_angles)
    log_sum_exp = max_beta + np.log(np.sum(np.exp(beta_angles - max_beta)))
    soft_max = log_sum_exp / beta
    
    # Gradient computation
    # d(soft_max)/d(config[j]) = (1/beta) * sum_i exp(beta*angles[i]) / sum_i exp(beta*angles[i]) * d(angles[i])/d(config[j])
    # d(angles[i])/d(config[j]) = -1/sqrt(1-max_dots[i]^2) * d(max_dots[i])/d(config[j])
    # d(max_dots[i])/d(config[j]) = grid[i] if j = argmax(dots[i,:]), else 0
    
    # For efficiency, use the soft weights
    weights = np.exp(beta_angles - max_beta) / np.sum(np.exp(beta_angles - max_beta))  # (M,)
    
    # For each grid point i, the nearest config point index
    nearest_idx = np.argmax(dots, axis=1)  # (M,)
    
    # d(angles[i])/d(config[j]) = -grid[i] / sqrt(1 - max_dots[i]^2) if j == nearest_idx[i]
    # = 0 otherwise
    
    # Build gradient
    grad = np.zeros_like(config)
    for i in range(M):
        w = weights[i]
        j = nearest_idx[i]
        md = max_dots[i]
        denom = np.sqrt(max(1e-15, 1.0 - md * md))
        grad[j] -= (w / beta) * grid[i] / denom
    
    # Convert to degrees
    value_deg = np.degrees(soft_max)
    grad_deg = grad * (180.0 / np.pi)
    
    return value_deg, grad_deg

def _soft_max_covering_radius_and_grad(config, beta):
    """
    Compute soft-max covering radius and its gradient.
    """
    M = 10000
    grid = _fib_sample(M)
    
    dots = grid @ config.T  # (M, N)
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
    
    beta_angles = beta * angles
    max_beta = np.max(beta_angles)
    exp_vals = np.exp(beta_angles - max_beta)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_beta + np.log(sum_exp)
    soft_max = log_sum_exp / beta
    
    weights = exp_vals / sum_exp
    nearest_idx = np.argmax(dots, axis=1)
    
    grad = np.zeros_like(config)
    for i in range(M):
        w = weights[i]
        j = nearest_idx[i]
        md = max_dots[i]
        denom = np.sqrt(max(1e-15, 1.0 - md * md))
        grad[j] -= (w / beta) * grid[i] / denom
    
    value_deg = np.degrees(soft_max)
    grad_deg = grad * (180.0 / np.pi)
    
    return value_deg, grad_deg

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    return _normalize_batch(pts)

def _lbfgs_stage(config, deadline, beta_start=50.0, beta_end=200.0, n_stages=3):
    """
    Run L-BFGS-B optimization with increasing beta for soft-max approximation.
    """
    current_config = config.copy()
    
    for stage in range(n_stages):
        if time.time() > deadline - 15:
            break
        
        beta = beta_start + (beta_end - beta_start) * stage / max(1, n_stages - 1)
        
        def objective(params):
            pts = _project_to_sphere(params.reshape(N, 3))
            val, _ = _soft_max_covering_radius_and_grad(pts, beta)
            return val
        
        def objective_and_grad(params):
            pts = _project_to_sphere(params.reshape(N, 3))
            val, grad = _soft_max_covering_radius_and_grad(pts, beta)
            # Project gradient to tangent space
            # For each point, remove radial component
            grad_tangent = np.zeros_like(pts)
            for i in range(N):
                radial = np.dot(grad[i], pts[i]) * pts[i]
                grad_tangent[i] = grad[i] - radial
            return val, grad_tangent.flatten()
        
        x0 = current_config.flatten()
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
            if result.success or result.fun < objective(x0):
                new_config = _project_to_sphere(result.x.reshape(N, 3))
                # Verify improvement
                val_new, _ = _soft_max_covering_radius_and_grad(new_config, beta)
                val_old, _ = _soft_max_covering_radius_and_grad(current_config, beta)
                if val_new < val_old:
                    current_config = new_config
        except Exception:
            pass
    
    return current_config

def _local_polish(config, deadline):
    """
    Final high-precision polish using exact covering radius evaluation.
    """
    current_config = config.copy()
    current_score = _covering_radius_hull(current_config)
    best_config = current_config.copy()
    best_score = current_score
    
    # Use a fine grid for local search
    M = 20000
    grid = _fib_sample(M)
    
    step_size = 0.01
    stagnation = 0
    
    while time.time() < deadline - 5:
        # Find worst region
        dots = grid @ current_config.T
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
        worst_idx = np.argmax(angles)
        worst_point = grid[worst_idx]
        worst_angle = angles[worst_idx]
        
        # Find nearest config point to worst point
        dots_worst = np.clip(worst_point @ current_config.T, -1.0, 1.0)
        nearest_idx = np.argmax(dots_worst)
        
        # Move nearest point towards worst point
        v = worst_point - current_config[nearest_idx]
        v_proj = v - np.dot(v, current_config[nearest_idx]) * current_config[nearest_idx]
        v_norm = np.linalg.norm(v_proj)
        
        if v_norm < 1e-10:
            break
        
        new_pt = current_config[nearest_idx] + step_size * v_proj
        new_pt = _normalize(new_pt)
        old_pt = current_config[nearest_idx].copy()
        current_config[nearest_idx] = new_pt
        
        new_score = _covering_radius_hull(current_config)
        if new_score < current_score:
            current_score = new_score
            stagnation = 0
            if new_score < best_score:
                best_score = new_score
                best_config = current_config.copy()
        else:
            current_config[nearest_idx] = old_pt
            stagnation += 1
            step_size *= 0.85
            if step_size < 0.001 or stagnation > 30:
                break
    
    return best_config

def _optimize(config, deadline):
    """
    Optimize the configuration using Lloyd's algorithm, then L-BFGS-B with
    soft-max surrogate, then local polish.
    """
    start_time = time.time()
    config = config.copy()
    
    # Phase 1: Lloyd's algorithm for good initial configuration
    M = 5000
    grid = _fib_sample(M)
    
    def lloyd_step(cfg):
        dots = grid @ cfg.T
        max_idx = np.argmax(dots, axis=1)
        new_cfg = np.zeros_like(cfg)
        for i in range(N):
            mask = (max_idx == i)
            count = np.sum(mask)
            if count > 0:
                center = np.mean(grid[mask], axis=0)
                new_cfg[i] = _normalize(center)
            else:
                new_cfg[i] = _normalize(np.random.randn(3))
        return new_cfg
    
    for _ in range(50):
        if time.time() > deadline - 60:
            break
        new_config = lloyd_step(config)
        if np.allclose(config, new_config, atol=1e-7):
            break
        config = new_config
    
    # Phase 2: L-BFGS-B with soft-max surrogate
    if time.time() < deadline - 30:
        config = _lbfgs_stage(config, deadline, beta_start=50.0, beta_end=200.0, n_stages=3)
    
    # Phase 3: Local polish with exact evaluation
    if time.time() < deadline - 10:
        config = _local_polish(config, deadline)
    
    return config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 seconds safety margin
    
    # Start with Fibonacci spiral
    config = _fib_sample(N)
    
    # Run optimization
    config = _optimize(config, deadline)
    
    # Final normalization and validation
    config = np.array([_normalize(p) for p in config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
    
    return config
