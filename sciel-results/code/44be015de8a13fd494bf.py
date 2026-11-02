import numpy as np
import time
from scipy.spatial import Voronoi

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill
START_TIME = time.time()

# Precompute indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_iu)

def _elapsed():
    return time.time() - START_TIME

def _deadline_exceeded(margin=5.0):
    return _elapsed() > DEADLINE - margin

def _compute_radius(c):
    """Compute the exact minimum radius."""
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(np.abs(c) > 0.5):
        return 0.0
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute the soft-min radius surrogate and its gradient.
    We want to maximize the minimum distance.
    Surrogate: -t * log( sum( exp(-d/2t) ) + sum( exp(-w/t) ) )
    Gradient is computed analytically.
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-15)
    
    # Pairwise terms: exp(-dists / (2*t))
    pair_terms = np.exp(-dists / (2.0 * t))
    
    # Wall terms: exp(-wall_dist / t)
    wall_dists = 0.5 - np.abs(c)
    wall_terms = np.exp(-wall_dists / t)
    
    total_sum = np.sum(pair_terms) + np.sum(wall_terms)
    
    if total_sum < 1e-300:
        return 0.0, np.zeros_like(c)
        
    softmin_val = -t * np.log(total_sum)
    
    grad_c = np.zeros_like(c)
    
    # Gradient of pairwise terms
    # d/dc_i ( -t * log(sum) ) = -t * (1/sum) * sum_j (d/dc_i exp(-d_ij/2t))
    # d/dc_i exp(-d_ij/2t) = exp(-d_ij/2t) * (-1/2t) * (c_i - c_j)/d_ij
    # Summing over j and i, and factoring out -t * (1/sum):
    # Term for pair (i,j) in grad[i]: (1/sum) * exp(-d_ij/2t) * (1/2) * (c_i - c_j)/d_ij
    # Term for pair (i,j) in grad[j]: (1/sum) * exp(-d_ij/2t) * (1/2) * (c_j - c_i)/d_ij
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # Weights for pairs
    pair_weights = pair_terms / total_sum
    
    # Contribution to grad[i] from pair (i,j) is +0.5 * w * unit_vec
    # Contribution to grad[j] from pair (i,j) is -0.5 * w * unit_vec
    pair_grad = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_c, _iu, pair_grad)
    np.add.at(grad_c, _ju, -pair_grad)
    
    # Gradient of wall terms
    # d/dc_ik exp(-w_ik/t) = exp(-w_ik/t) * (-1/t) * (-sign(c_ik))
    # Summing and factoring -t * (1/sum):
    # Term is (1/sum) * exp(-w_ik/t) * sign(c_ik)
    
    wall_weights = wall_terms / total_sum
    signs = np.sign(c)
    grad_c += wall_weights * signs
    
    return softmin_val, grad_c

def _lloyd_step(c, step_size=1.0):
    """Perform one Lloyd relaxation step."""
    try:
        vor = Voronoi(c)
    except Exception:
        return c
    
    c_new = c.copy()
    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if len(region) == 0 or -1 in region:
            continue
        verts = vor.vertices[region]
        if len(verts) == 0:
            continue
        centroid = np.mean(verts, axis=0)
        # Clip centroid to be within a slightly smaller cube to avoid boundary issues
        centroid = np.clip(centroid, -0.49, 0.49)
        c_new[i] = c[i] + step_size * (centroid - c[i])
        c_new[i] = np.clip(c_new[i], -0.499999, 0.499999)
    return c_new

def _optimize_phase(c_init, deadline, max_lloyd=20, max_grad=200):
    """
    Optimize a starting configuration using Lloyd relaxation and Gradient Descent.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Phase 1: Lloyd Relaxation
    for i in range(max_lloyd):
        if _deadline_exceeded(margin=10.0):
            break
        c = _lloyd_step(c, step_size=0.5)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    # Phase 2: Gradient Descent on Soft-Min
    # Anneal temperature t from high to low
    t_start = 0.1
    t_end = 0.005
    num_temp_steps = 10
    
    for k in range(num_temp_steps):
        if _deadline_exceeded(margin=10.0):
            break
        t = t_start * (t_end / t_start) ** (k / (num_temp_steps - 1))
        
        # Run a few gradient steps at this temperature
        for step in range(max_grad // num_temp_steps):
            if _deadline_exceeded(margin=10.0):
                break
            
            val, grad = _softmin_radius_and_grad(c, t)
            
            # Simple gradient descent with adaptive step size
            # We want to move in the direction of -grad to minimize val (maximize radius)
            # Normalize gradient to control step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Heuristic step size: start small, decrease if radius drops
            step_size = 0.01 * (1.0 + 5.0 * (1.0 - k / num_temp_steps))
            
            c_trial = c - step_size * (grad / grad_norm)
            c_trial = np.clip(c_trial, -0.499999, 0.499999)
            
            r_trial = _compute_radius(c_trial)
            
            if r_trial > best_r:
                best_r = r_trial
                best_c = c_trial.copy()
            
            # Accept if it improves the soft-min objective or with some probability
            # For simplicity, accept if radius improves, or if soft-min improves
            val_trial, _ = _softmin_radius_and_grad(c_trial, t)
            
            # Metropolis-like acceptance on the soft-min value
            delta_val = val_trial - val
            if delta_val < 0 or np.random.random() < np.exp(-delta_val / t):
                c = c_trial
            else:
                # If rejected, try a smaller step? Or just stop if stuck?
                # Let's try a smaller step for a few iterations
                pass
    
    return best_c, best_r

def _seed_fcc():
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    return points

def _seed_grid():
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _seed_bcc():
    m = 3
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                points.append([x, y, z])
                points.append([x + 0.5*spacing, y + 0.5*spacing, z + 0.5*spacing])
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    return points

def _seed_random():
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_c = None
    best_r = 0.0
    
    # List of initialization strategies
    seeds = [
        _seed_fcc,
        _seed_grid,
        _seed_bcc,
        _seed_random
    ]
    
    # We will iterate through seeds, allocating time based on remaining budget
    # If we have plenty of time, we can run multiple iterations per seed
    # Otherwise, we just try each seed once
    
    # Estimate time per optimization cycle
    # Lloyd is fast, Gradient is slower.
    # Let's try to run each seed at least once if time permits.
    
    for seed_func in seeds:
        if _deadline_exceeded(margin=15.0):
            break
        
        c_init = seed_func()
        c_opt, r_opt = _optimize_phase(c_init, DEADLINE)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # If we still have time, try to polish the best result further
    if best_c is not None and not _deadline_exceeded(margin=20.0):
        # Run another round of optimization on the best result
        c_opt, r_opt = _optimize_phase(best_c, DEADLINE, max_lloyd=10, max_grad=100)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    if best_c is None:
        # Fallback
        best_c = _seed_random()
        
    return best_c
