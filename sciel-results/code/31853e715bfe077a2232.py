# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def smooth_min_max_ratio(points, beta=50.0):
    """
    Smooth surrogate for (d_min/d_max)^2 using log-sum-exp.
    """
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    
    # Soft min via log-sum-exp
    neg_dists = -dists / beta
    max_neg = np.max(neg_dists)
    soft_min = -beta * (max_neg + np.log(np.sum(np.exp(neg_dists - max_neg))))
    
    # Soft max via log-sum-exp  
    pos_dists = dists / beta
    max_pos = np.max(pos_dists)
    soft_max = beta * (max_pos + np.log(np.sum(np.exp(pos_dists - max_pos))))
    
    if soft_max <= 0:
        return 0.0
    
    ratio = soft_min / soft_max
    return ratio * ratio

def nelder_mead_polish(points, time_limit, beta=50.0, seed=0):
    """
    Nelder-Mead optimization on the smooth surrogate objective.
    """
    from scipy.optimize import minimize
    
    n = points.shape[0]
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Minimize negative of smooth score
        return -smooth_min_max_ratio(pts, beta=beta)
    
    start_time = time.time()
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return points.copy()

def local_search_smooth(points, time_limit, beta=50.0, seed=0):
    """
    Local search with adaptive step size on smooth surrogate.
    """
    n = points.shape[0]
    current = points.copy()
    current_score = smooth_min_max_ratio(current, beta=beta)
    best = current.copy()
    best_score = current_score
    
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = rng.randint(n)
        
        # Generate perturbation
        perturbation = rng.randn(2) * step_size
        
        new_point = current[idx].copy() + perturbation
        trial = current.copy()
        trial[idx] = new_point
        
        trial_score = smooth_min_max_ratio(trial, beta=beta)
        
        # Acceptance criterion for smooth objective
        if trial_score > current_score:
            current = trial
            current_score = trial_score
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        else:
            no_improve_count += 1
        
        # Adapt step size
        if no_improve_count > max_no_improve:
            step_size *= 0.8
            no_improve_count = 0
            if step_size < 1e-8:
                break
        elif no_improve_count > max_no_improve // 2:
            step_size *= 0.95
    
    return best, best_score

def optimize_geometry(points, time_limit, seed=0):
    """
    Combined optimization: local search with smooth surrogate + Nelder-Mead polish.
    """
    start_time = time.time()
    
    # Phase 1: Local search with smooth surrogate
    time_for_local = time_limit * 0.7
    local_points, local_score = local_search_smooth(points, time_for_local, beta=50.0, seed=seed)
    
    # Phase 2: Nelder-Mead polish on the result
    remaining = time_limit - (time.time() - start_time) - 2
    if remaining > 5:
        polished_points = nelder_mead_polish(local_points, remaining, beta=50.0, seed=seed)
        polished_score = calculate_score(polished_points)
        
        if polished_score > local_score:
            return polished_points, polished_score
    
    return local_points, local_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    d = 2
    
    global_deadline = time.time() + 620  # Safety margin
    
    # Initial candidates
    candidates = []
    
    # 1. Hexagonal lattice (best known packing structure)
    # Generate hex grid points
    hex_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q + 0.5 * r
            y = np.sqrt(3) / 2 * r
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    # Select 16 points closest to center
    dists_to_origin = np.linalg.norm(hex_points, axis=1)
    idx = np.argsort(dists_to_origin)[:16]
    candidates.append(hex_points[idx])
    
    # 2. 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    candidates.append(np.column_stack((xx.ravel(), yy.ravel())))
    
    # 3. Circle with two rings
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(np.pi / 8, 2 * np.pi, 8, endpoint=False)
    r1, r2 = 1.0, 0.5
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    candidates.append(np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])]))
    
    # 4. Random points in a disk
    rng_init = np.random.RandomState(42)
    angles = rng_init.rand(n) * 2 * np.pi
    radii = np.sqrt(rng_init.rand(n)) * 1.5
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    candidates.append(np.column_stack((x, y)))
    
    # 5. Random points in a square
    rng_init = np.random.RandomState(123)
    candidates.append(rng_init.rand(n, 2) * 3 - 1.5)
    
    # Optimize each candidate
    best_points = None
    best_score = -1
    
    time_per_candidate = 100
    
    for i, cand in enumerate(candidates):
        if time.time() > global_deadline - 10:
            break
        
        cand = cand.astype(float)
        res_points, res_score = optimize_geometry(cand, time_per_candidate, seed=1000 + i)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # If time remains, do a final refinement on the best candidate
    if best_points is not None and time.time() < global_deadline - 5:
        remaining = global_deadline - 5 - time.time()
        if remaining > 10:
            res_points, res_score = optimize_geometry(best_points, remaining, seed=9999)
            if res_score > best_score:
                best_points = res_points
                best_score = res_score
    
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))
    
    return best_points

# EVOLVE-BLOCK-END
