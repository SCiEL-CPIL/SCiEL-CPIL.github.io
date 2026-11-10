import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 115.0  # Safety margin from 120s

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns (radius, is_valid).
    """
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    if np.any(bound_dists < -1e-9):
        return -1.0, False
        
    min_bound = np.min(bound_dists)
    
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r, True

def make_hex_init(noise_scale=0.01):
    """Create a jittered hexagonal lattice initialization."""
    s = 1.0
    points = []
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    hex_pos = points[indices].copy()
    max_abs = np.max(np.abs(hex_pos))
    if max_abs > 0:
        hex_pos = hex_pos * (0.49 / max_abs)
    hex_pos += np.random.normal(0, noise_scale, size=hex_pos.shape)
    hex_pos = np.clip(hex_pos, -0.4999, 0.4999)
    return hex_pos

def objective_min_dist(x):
    """
    Objective: minimize the negative of the minimum distance (pairwise and boundary).
    We want to maximize the minimum of:
      - pairwise distances
      - distance to boundary for each point
    So we minimize -min(min_pair_dist, min_bound_dist).
    
    Actually, let's define:
      d_pair = minimum pairwise distance
      d_bound = minimum distance to boundary (0.5 - max(abs(coord)))
      r = min(d_pair/2, d_bound)
    We want to maximize r, so minimize -r.
    """
    positions = x.reshape(N, 2)
    
    # Boundary distances
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    min_bound = np.min(bound_dists)
    
    # Pairwise distances using cdist (C-optimized)
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_bound, min_dist / 2.0)
    
    return -r

def construct():
    start_time = time.time()
    
    best_radius = 0.0
    best_positions = None
    
    # Generate multiple hexagonal initializations with different noise levels
    inits = []
    for noise in [0.005, 0.01, 0.02, 0.03]:
        inits.append(make_hex_init(noise))
    
    # Also add a random init
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # Phase 1: Use Nelder-Mead to maximize the minimum radius directly
    for init_pos in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        x0 = init_pos.flatten()
        
        try:
            # Nelder-Mead for direct optimization of min radius
            # Use moderate initial simplex size
            res = minimize(
                objective_min_dist, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'maxfev': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10
                }
            )
        except Exception:
            continue
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        new_pos = res.x.reshape(N, 2)
        r, valid = compute_radius(new_pos)
        
        if valid and r > best_radius:
            best_radius = r
            best_positions = new_pos.copy()
    
    # Phase 2: Refine with Powell on a penalty-based objective for the found radius
    if best_positions is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 30:
            # Try to push the radius slightly higher using penalty method
            R_target = best_radius * 1.005
            x0 = best_positions.flatten()
            
            def penalty_obj(x, R):
                positions = x.reshape(N, 2)
                
                # Boundary penalty
                max_abs = np.max(np.abs(positions), axis=1)
                bound_dists = 0.5 - max_abs
                boundary_violations = np.maximum(0, R - bound_dists)
                penalty_bound = np.sum(boundary_violations**2)
                
                # Pairwise penalty using cdist
                dists = cdist(positions, positions)
                np.fill_diagonal(dists, np.inf)
                pair_dists = dists[np.triu_indices(N, k=1)]
                violations = np.maximum(0, 2*R - pair_dists)
                penalty_pairs = np.sum(violations**2)
                
                return penalty_bound + penalty_pairs
            
            def obj(x):
                return penalty_obj(x, R_target)
            
            try:
                remaining = DEADLINE - (time.time() - start_time)
                maxiter = max(100, int(remaining * 5))
                res = minimize(
                    obj, x0, method='Powell',
                    options={'maxiter': maxiter, 'maxfev': maxiter * 3}
                )
            except Exception:
                res = None
            
            if res is not None:
                elapsed = time.time() - start_time
                if elapsed < DEADLINE - 5:
                    new_pos = res.x.reshape(N, 2)
                    r, valid = compute_radius(new_pos)
                    if valid and r > best_radius:
                        best_radius = r
                        best_positions = new_pos.copy()
    
    # Phase 3: Final refinement with L-BFGS-B on smooth approximation if time permits
    if best_positions is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            # Use a soft-min approximation for smoother optimization
            def soft_min_obj(x):
                positions = x.reshape(N, 2)
                
                # Boundary distances
                max_abs = np.max(np.abs(positions), axis=1)
                bound_dists = 0.5 - max_abs
                
                # Pairwise distances
                dists = cdist(positions, positions)
                np.fill_diagonal(dists, np.inf)
                
                # Soft minimum of all distances (pairwise/2 and boundary)
                # We want to maximize the minimum, so minimize -soft_min
                # soft_min = -beta * log(sum(exp(-d/beta)))
                # Use a moderate beta for smoothness
                beta = 0.01
                
                # Combine all distances: pairwise/2 and boundary distances
                pair_dists_half = dists[np.triu_indices(N, k=1)] / 2.0
                all_dists = np.concatenate([pair_dists_half, bound_dists])
                
                # Soft min
                log_sum_exp = np.log(np.sum(np.exp(-all_dists / beta)))
                soft_min = -beta * log_sum_exp
                
                return -soft_min
            
            x0 = best_positions.flatten()
            
            try:
                res = minimize(
                    soft_min_obj, x0, method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 200, 'maxls': 50}
                )
            except Exception:
                res = None
            
            if res is not None:
                new_pos = res.x.reshape(N, 2)
                r, valid = compute_radius(new_pos)
                if valid and r > best_radius:
                    best_radius = r
                    best_positions = new_pos.copy()
    
    # Fallback: return hex init if nothing else worked
    if best_positions is None:
        best_positions = make_hex_init(0.01)
    
    return best_positions
