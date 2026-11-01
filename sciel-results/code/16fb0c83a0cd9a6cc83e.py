import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _normalize_batch(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fib_sample(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_hull(config):
    """
    Compute covering radius in degrees using convex hull facets.
    Returns (radius_deg, circumcenters_array)
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([])
    
    simplices = hull.simplices
    n_facets = len(simplices)
    
    p0 = config[simplices[:, 0]]
    p1 = config[simplices[:, 1]]
    p2 = config[simplices[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    
    c = np.cross(v1, v2)
    c_norms = np.linalg.norm(c, axis=1)
    
    valid_mask = c_norms > 1e-12
    if not np.any(valid_mask):
        return 0.0, np.array([])
    
    c = c[valid_mask]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    valid_facets = simplices[valid_mask]
    
    dots0 = np.einsum('ij,ij->i', c, p0[valid_mask])
    dots1 = np.einsum('ij,ij->i', c, p1[valid_mask])
    dots2 = np.einsum('ij,ij->i', c, p2[valid_mask])
    
    dots = np.column_stack([dots0, dots1, dots2])
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    angle_spread = np.max(angles, axis=1) - np.min(angles, axis=1)
    valid_circ = angle_spread < 1e-6
    
    if not np.any(valid_circ):
        grid = _fib_sample(10000)
        dots_grid = grid @ config.T
        min_dots = np.min(dots_grid, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([])
    
    c = c[valid_circ]
    
    dots_all = c @ config.T
    dots_all = np.clip(dots_all, -1.0, 1.0)
    max_dots = np.max(dots_all, axis=1)
    min_angles = np.arccos(max_dots)
    
    max_angle = np.max(min_angles)
    radius_deg = np.degrees(max_angle)
    
    return radius_deg, c

def _soft_max_surrogate(config, beta=50.0):
    """
    Smooth surrogate for the covering radius using soft-max over circumcenters.
    Returns (surrogate_value, circumcenters, angles)
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([]), np.array([])
    
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
        return 0.0, np.array([]), np.array([])
    
    c = c[valid_mask]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    
    dots0 = np.einsum('ij,ij->i', c, p0[valid_mask])
    dots1 = np.einsum('ij,ij->i', c, p1[valid_mask])
    dots2 = np.einsum('ij,ij->i', c, p2[valid_mask])
    
    dots = np.column_stack([dots0, dots1, dots2])
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    angle_spread = np.max(angles, axis=1) - np.min(angles, axis=1)
    valid_circ = angle_spread < 1e-6
    
    if not np.any(valid_circ):
        grid = _fib_sample(10000)
        dots_grid = grid @ config.T
        min_dots = np.min(dots_grid, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([]), np.array([])
    
    c = c[valid_circ]
    
    dots_all = c @ config.T
    dots_all = np.clip(dots_all, -1.0, 1.0)
    min_angles = np.arccos(np.max(dots_all, axis=1))
    
    # Soft-max surrogate
    scaled = beta * min_angles
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    soft_max = (max_scaled + np.log(np.sum(exp_scaled))) / beta
    
    return soft_max, c, min_angles

def _surrogate_grad(config, beta=50.0):
    """
    Compute gradient of the soft-max surrogate with respect to config points.
    Uses finite differences for robustness.
    """
    val, c, min_angles = _soft_max_surrogate(config, beta)
    
    if len(c) == 0:
        return np.zeros_like(config)
    
    # Compute weights for each circumcenter
    scaled = beta * min_angles
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # For each circumcenter, find the nearest config point
    dots_all = c @ config.T
    dots_all = np.clip(dots_all, -1.0, 1.0)
    nearest_idx = np.argmax(dots_all, axis=1)
    
    grad = np.zeros_like(config)
    
    for i in range(len(c)):
        w = weights[i]
        if w < 1e-10:
            continue
        idx = nearest_idx[i]
        cc = c[i]
        
        # Gradient of arccos(max_j dot(c, p_j)) w.r.t. p_idx
        # d/dp arccos(dot(c,p)) = -c / sqrt(1 - dot(c,p)^2)
        dot_val = np.dot(cc, config[idx])
        dot_val = np.clip(dot_val, -1.0 + 1e-10, 1.0 - 1e-10)
        denom = np.sqrt(1.0 - dot_val * dot_val)
        
        if denom < 1e-10:
            continue
        
        grad[idx] -= w * cc / denom
    
    return grad

def _lloyd_step(cfg, grid):
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

def _local_optimize(config, deadline, min_time_remaining=10.0):
    """
    Local optimization using circumcenter-based critical points with adaptive steps.
    """
    config = config.copy()
    
    current_score, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.04
    stagnation = 0
    max_stagnation = 40
    
    while time.time() < deadline - min_time_remaining:
        score, circumcenters = _covering_radius_hull(config)
        
        if len(circumcenters) == 0:
            break
        
        current_score = score
        
        dots_all = circumcenters @ config.T
        dots_all = np.clip(dots_all, -1.0, 1.0)
        max_dots = np.max(dots_all, axis=1)
        min_angles = np.arccos(max_dots)
        
        worst_idx = np.argmax(min_angles)
        worst_point = circumcenters[worst_idx]
        worst_angle = min_angles[worst_idx]
        
        dots_to_worst = np.clip(worst_point @ config.T, -1.0, 1.0)
        sorted_idx = np.argsort(-dots_to_worst)
        
        improved = False
        
        for trial in range(4):
            if trial >= len(sorted_idx):
                break
            idx = sorted_idx[trial]
            
            v = worst_point - config[idx]
            v_proj = v - np.dot(v, config[idx]) * config[idx]
            v_norm = np.linalg.norm(v_proj)
            
            if v_norm < 1e-10:
                continue
            
            v_dir = v_proj / v_norm
            
            for ss in [step_size, step_size * 0.5, step_size * 1.5, step_size * 0.25]:
                new_pt = config[idx] + ss * v_dir
                new_pt = _normalize(new_pt)
                
                old_pt = config[idx].copy()
                config[idx] = new_pt
                
                new_score, _ = _covering_radius_hull(config)
                
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    stagnation = 0
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                    if new_score < current_score - 0.01:
                        step_size = min(step_size * 1.15, 0.1)
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

def _gradient_optimize(config, deadline, min_time_remaining=15.0):
    """
    Gradient-based optimization using the soft-max surrogate.
    """
    config = config.copy()
    current_score, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    lr = 0.01
    stagnation = 0
    
    while time.time() < deadline - min_time_remaining:
        score, _ = _covering_radius_hull(config)
        current_score = score
        
        if current_score < best_score:
            best_score = current_score
            best_config = config.copy()
            stagnation = 0
        else:
            stagnation += 1
        
        if stagnation > 30:
            break
        
        grad = _surrogate_grad(config, beta=50.0)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad = grad / grad_norm
        
        # Update with projection to sphere
        new_config = config - lr * grad
        new_config = _normalize_batch(new_config)
        
        # Check if improvement
        new_score, _ = _covering_radius_hull(new_config)
        
        if new_score < current_score - 1e-10:
            config = new_config
            if new_score < best_score:
                best_score = new_score
                best_config = config.copy()
                lr = min(lr * 1.1, 0.05)
        else:
            lr *= 0.8
            if lr < 0.001:
                break
    
    return best_config

def _polish(config, deadline, min_time_remaining=5.0):
    """
    High-precision polish with very small steps.
    """
    config = config.copy()
    current_score, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.005
    stagnation = 0
    
    while time.time() < deadline - min_time_remaining:
        score, circumcenters = _covering_radius_hull(config)
        
        if len(circumcenters) == 0:
            break
        
        current_score = score
        
        dots_all = circumcenters @ config.T
        dots_all = np.clip(dots_all, -1.0, 1.0)
        max_dots = np.max(dots_all, axis=1)
        min_angles = np.arccos(max_dots)
        
        threshold = np.max(min_angles) - 0.001
        worst_mask = min_angles > threshold
        worst_indices = np.where(worst_mask)[0]
        
        if len(worst_indices) == 0:
            break
        
        improved = False
        
        for wi in worst_indices[:6]:
            worst_point = circumcenters[wi]
            
            dots_to_worst = np.clip(worst_point @ config.T, -1.0, 1.0)
            sorted_idx = np.argsort(-dots_to_worst)
            
            for trial in range(3):
                if trial >= len(sorted_idx):
                    break
                idx = sorted_idx[trial]
                
                v = worst_point - config[idx]
                v_proj = v - np.dot(v, config[idx]) * config[idx]
                v_norm = np.linalg.norm(v_proj)
                
                if v_norm < 1e-10:
                    continue
                
                v_dir = v_proj / v_norm
                
                for ss in [step_size, step_size * 0.5]:
                    new_pt = config[idx] + ss * v_dir
                    new_pt = _normalize(new_pt)
                    
                    old_pt = config[idx].copy()
                    config[idx] = new_pt
                    
                    new_score, _ = _covering_radius_hull(config)
                    
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
            
            if improved:
                break
        
        if not improved:
            stagnation += 1
            step_size *= 0.9
            if step_size < 0.0005 or stagnation > 60:
                break
    
    return best_config

def _optimize(config, deadline):
    """
    Full optimization pipeline.
    """
    config = config.copy()
    
    # Phase 1: Lloyd's algorithm
    M = 10000
    grid = _fib_sample(M)
    
    for iteration in range(200):
        if time.time() > deadline - 50:
            break
        new_config = _lloyd_step(config, grid)
        if np.allclose(config, new_config, atol=1e-8):
            break
        config = new_config
    
    # Phase 2: Local optimization
    if time.time() < deadline - 30:
        config = _local_optimize(config, deadline, min_time_remaining=25.0)
    
    # Phase 3: Gradient-based optimization
    if time.time() < deadline - 15:
        config = _gradient_optimize(config, deadline, min_time_remaining=12.0)
    
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
