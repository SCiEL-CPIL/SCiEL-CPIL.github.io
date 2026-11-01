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

def _get_circumcenters_and_radii(config):
    try:
        hull = ConvexHull(config)
    except Exception:
        return np.array([]), np.array([]), np.array([], dtype=int)
    
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
        return np.array([]), np.array([]), np.array([], dtype=int)
        
    c = c[valid_mask]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    
    dots = c @ config.T
    dots = np.clip(dots, -1.0, 1.0)
    nearest_idx = np.argmax(dots, axis=1)
    nearest_dots = dots[np.arange(len(c)), nearest_idx]
    radii = np.arccos(nearest_dots)
    
    return c, radii, nearest_idx

def _covering_radius(config):
    c, radii, _ = _get_circumcenters_and_radii(config)
    if len(c) == 0:
        grid = _fib_sample(10000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.max(max_angle)
    return np.max(radii)

def _soft_max_radius(radii, tau):
    if len(radii) == 0:
        return 1.0
    shifted = radii - np.max(radii)
    return np.max(radii) + tau * np.log(np.sum(np.exp(shifted / tau)))

def _soft_max_radius_and_grad(radii, tau):
    if len(radii) == 0:
        return 1.0, np.zeros_like(radii)
    shifted = radii - np.max(radii)
    weights = np.exp(shifted / tau)
    weights /= np.sum(weights)
    val = np.max(radii) + tau * np.log(np.sum(weights))
    grad = weights
    return val, grad

def _project_to_sphere(x_flat):
    pts = x_flat.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _smooth_objective(x_flat, tau=0.05):
    pts = _project_to_sphere(x_flat)
    c, radii, _ = _get_circumcenters_and_radii(pts)
    if len(c) == 0:
        return 1.0
    return _soft_max_radius(radii, tau)

def _smooth_gradient(x_flat, tau=0.05, eps=1e-6):
    pts = _project_to_sphere(x_flat)
    grad = np.zeros_like(x_flat)
    
    for i in range(3 * N):
        x_plus = x_flat.copy()
        x_plus[i] += eps
        pts_plus = _project_to_sphere(x_plus)
        
        x_minus = x_flat.copy()
        x_minus[i] -= eps
        pts_minus = _project_to_sphere(x_minus)
        
        c_plus, radii_plus, _ = _get_circumcenters_and_radii(pts_plus)
        c_minus, radii_minus, _ = _get_circumcenters_and_radii(pts_minus)
        
        if len(c_plus) == 0 or len(c_minus) == 0:
            continue
        
        f_plus = _soft_max_radius(radii_plus, tau)
        f_minus = _soft_max_radius(radii_minus, tau)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad

def _repulsive_force_optimize(config, deadline, min_time_remaining=10.0):
    config = config.copy()
    c, radii, nearest_idx = _get_circumcenters_and_radii(config)
    if len(c) == 0:
        return config
        
    current_R = np.max(radii)
    best_config = config.copy()
    best_R = current_R
    
    step_size = 0.01
    stagnation = 0
    max_stagnation = 150
    
    while time.time() < deadline - min_time_remaining:
        c, radii, nearest_idx = _get_circumcenters_and_radii(config)
        if len(c) == 0:
            break
            
        current_R = np.max(radii)
        
        if current_R < best_R - 1e-10:
            best_R = current_R
            best_config = config.copy()
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                break
                
        # Compute forces from circumcenters (push towards worst circumcenters)
        forces = np.zeros_like(config)
        weights = np.exp(-100.0 * (current_R - radii))
        
        for k in range(len(c)):
            i = nearest_idx[k]
            v = c[k] - config[i]
            v_proj = v - np.dot(v, config[i]) * config[i]
            forces[i] += weights[k] * v_proj
        
        # Add repulsive forces between points to prevent clustering
        for i in range(N):
            for j in range(i + 1, N):
                d = np.dot(config[i], config[j])
                d = np.clip(d, -1.0, 0.999)
                angle = np.arccos(d)
                if angle < 0.5:  # Only repel when close
                    repel_strength = 0.01 / (angle + 0.1)
                    v_ij = config[j] - config[i]
                    v_ij_proj_i = v_ij - np.dot(v_ij, config[i]) * config[i]
                    v_ij_proj_j = -v_ij_proj_i
                    forces[i] += repel_strength * v_ij_proj_i
                    forces[j] += repel_strength * v_ij_proj_j
        
        new_config = config.copy()
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-12:
                dir_vec = forces[i] / f_norm
                new_pt = new_config[i] + step_size * dir_vec
                new_config[i] = _normalize(new_pt)
        
        c_new, radii_new, _ = _get_circumcenters_and_radii(new_config)
        if len(c_new) == 0:
            break
        new_R = np.max(radii_new)
        
        if new_R < current_R - 1e-9:
            config = new_config
            current_R = new_R
            step_size = min(step_size * 1.02, 0.03)
        else:
            step_size *= 0.95
            if step_size < 0.001:
                break
                
    return best_config

def _polish(config, deadline, min_time_remaining=5.0):
    config = config.copy()
    c, radii, nearest_idx = _get_circumcenters_and_radii(config)
    if len(c) == 0:
        return config
        
    current_R = np.max(radii)
    best_config = config.copy()
    best_R = current_R
    
    step_size = 0.001
    stagnation = 0
    max_stagnation = 300
    
    while time.time() < deadline - min_time_remaining:
        c, radii, nearest_idx = _get_circumcenters_and_radii(config)
        if len(c) == 0:
            break
            
        current_R = np.max(radii)
        
        if current_R < best_R - 1e-12:
            best_R = current_R
            best_config = config.copy()
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                break
                
        forces = np.zeros_like(config)
        weights = np.exp(-300.0 * (current_R - radii))
        
        for k in range(len(c)):
            i = nearest_idx[k]
            v = c[k] - config[i]
            v_proj = v - np.dot(v, config[i]) * config[i]
            forces[i] += weights[k] * v_proj
        
        new_config = config.copy()
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-12:
                dir_vec = forces[i] / f_norm
                new_pt = new_config[i] + step_size * dir_vec
                new_config[i] = _normalize(new_pt)
        
        c_new, radii_new, _ = _get_circumcenters_and_radii(new_config)
        if len(c_new) == 0:
            break
        new_R = np.max(radii_new)
        
        if new_R < current_R - 1e-12:
            config = new_config
            current_R = new_R
        else:
            step_size *= 0.9
            if step_size < 0.00005:
                break
                
    return best_config

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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial configuration: Fibonacci spiral
    config = _fib_sample(N)
    
    # Phase 1: Lloyd's algorithm
    M = 10000
    grid = _fib_sample(M)
    
    for iteration in range(30):
        if time.time() > deadline - 90:
            break
        new_config = _lloyd_step(config, grid)
        if np.allclose(config, new_config, atol=1e-8):
            break
        config = new_config
        
    # Phase 2: L-BFGS-B with smooth objective
    if time.time() < deadline - 50:
        try:
            x0 = config.flatten()
            result = minimize(
                _smooth_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1500}
            )
            if result.fun < _smooth_objective(x0):
                config = _project_to_sphere(result.x)
        except Exception:
            pass
    
    # Phase 3: Repulsive force optimization
    if time.time() < deadline - 25:
        config = _repulsive_force_optimize(config, deadline, min_time_remaining=20.0)
        
    # Phase 4: Polish
    if time.time() < deadline - 5:
        config = _polish(config, deadline, min_time_remaining=3.0)
        
    # Final normalization
    config = _normalize_batch(config)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
                
    return config
