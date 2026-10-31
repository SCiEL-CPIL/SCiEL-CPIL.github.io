"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

N = 26

# EVOLVE-BLOCK-START

def _objective(x, penalty_weight=500.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 1e-6)
    
    # 1. Boundary Violations (Smooth Squared Penalty)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    
    boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                       np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # 2. Pairwise Overlap Violations (Vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = np.maximum(min_dist_sq, 1e-12)
    smooth_overlap = (violation ** 2) / (4.0 * denom)
    
    overlap_penalty = np.sum(smooth_overlap[mask])
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + penalty_weight * total_penalty

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    Uses a more aggressive approach that preserves more radius.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(100):
        valid = True
        
        # 1. Boundary fix - only shrink if necessary
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + 1e-9:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        # 2. Overlap fix - proportional shrink to minimize radius loss
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-9:
                    # Shrink both circles proportionally to their sizes
                    overlap = sum_r - dist
                    total_r = sum_r
                    shrink_i = (radii[i] / total_r) * overlap * 1.001
                    shrink_j = (radii[j] / total_r) * overlap * 1.001
                    radii[i] = max(radii[i] - shrink_i, 1e-5)
                    radii[j] = max(radii[j] - shrink_j, 1e-5)
                    valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    Uses a hexagonal-inspired pattern with corner circles for better density.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with corner circles (larger circles in corners)
    corner_positions = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    corner_radii = [0.14, 0.14, 0.14, 0.14]
    
    for i in range(min(4, N)):
        pos[i] = corner_positions[i]
        radii[i] = corner_radii[i]
    
    # Add a central circle
    if N > 4:
        pos[4] = [0.5, 0.5]
        radii[4] = 0.18
    
    # Fill remaining circles using best-fit
    for i in range(min(5, N), N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 7 + 13)
        
        # Sample more points with better coverage
        for _ in range(2000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            
            # Max radius allowed by existing circles
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.01:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.01:
            best_x, best_y = 0.5, 0.5
            best_r = 0.015
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.985
        
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.08, 0.92, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    # Force validity by shrinking
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _pairwise_constraints(x):
    """
    Compute pairwise distance constraints: dist_ij >= r_i + r_j
    Returns array of (dist_ij - r_i - r_j) for all pairs i < j
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    return (dist - min_dist)[mask]

def _boundary_constraints(x):
    """
    Compute boundary constraints: circle must be inside [0,1]^2
    Returns array of min distance to each wall for each circle
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # For each circle, the minimum distance to any wall
    # Must be >= radius
    left = pos[:, 0] - radii
    right = (1.0 - pos[:, 0]) - radii
    bottom = pos[:, 1] - radii
    top = (1.0 - pos[:, 1]) - radii
    
    # Return all 4*N values (each must be >= 0)
    return np.concatenate([left, right, bottom, top])

def _construct_with_trust_constr(x0, deadline):
    """
    Run trust-constr solver with explicit nonlinear constraints.
    Returns (pos, radii, score) or None if failed.
    """
    def objective_tc(x):
        return -np.sum(x[2*N:])
    
    # Nonlinear constraints
    pairwise_constraint = NonlinearConstraint(_pairwise_constraints, 0.0, np.inf)
    boundary_constraint = NonlinearConstraint(_boundary_constraints, 0.0, np.inf)
    
    constraints = [pairwise_constraint, boundary_constraint]
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
    
    try:
        result = minimize(objective_tc, x0, method='trust-constr',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 5000, 'gtol': 1e-9, 'ftol': 1e-12})
        
        if time.time() > deadline:
            return None
            
        if result.success or result.fun < objective_tc(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp, score
    except Exception:
        pass
        
    return None

def _polish_l_bfgs(x0, deadline):
    """
    Final high-precision polish using L-BFGS-B with tight tolerances.
    """
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    penalty_weight = 2000.0
    
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                          args=(penalty_weight,),
                          options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12})
        
        if time.time() > deadline:
            return None
            
        x_opt = result.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x_opt[2*N:], 1e-4)
        
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        score = np.sum(radii_pp)
        
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp, score
    except Exception:
        pass
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Strategy: Multi-start with different initializations
    # 1. Best-fit greedy (improved with corners)
    # 2. Random perturbations of best-fit
    # 3. Pure random (if time permits)
    
    inits = []
    
    # 1. Best Fit (improved)
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Perturbed Best Fit (4 variations for more diversity)
    for i in range(4):
        pos_p, rad_p = _initial_random(100 + i * 7)
        inits.append((pos_p, rad_p))
        
    # 3. Two more random with different seeds
    pos_r1, rad_r1 = _initial_random(999)
    inits.append((pos_r1, rad_r1))
    pos_r2, rad_r2 = _initial_random(1234)
    inits.append((pos_r2, rad_r2))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    penalty_weight = 500.0
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # First, try trust-constr with explicit constraints
        tc_result = _construct_with_trust_constr(x0, deadline)
        if tc_result is not None:
            pos_tc, radii_tc, score_tc = tc_result
            if score_tc > best_score:
                best_score = score_tc
                best_pos = pos_tc.copy()
                best_radii = radii_tc.copy()
        
        # Also try L-BFGS-B with penalty as backup
        if time.time() < deadline - 15:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_weight,),
                                  options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-9})
                
                if result.fun < _objective(x0, penalty_weight) - 1e-10:
                    x_opt = result.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                    
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
                        
            except Exception:
                pass
    
    # Final high-precision polish on the best result
    if time.time() < deadline - 8:
        x_best = np.concatenate([best_pos.flatten(), best_radii])
        polish_result = _polish_l_bfgs(x_best, deadline)
        if polish_result is not None:
            pos_pol, radii_pol, score_pol = polish_result
            if score_pol > best_score:
                best_score = score_pol
                best_pos = pos_pol.copy()
                best_radii = radii_pol.copy()
    
    # Second polish pass if time permits
    if time.time() < deadline - 3:
        x_best = np.concatenate([best_pos.flatten(), best_radii])
        polish_result2 = _polish_l_bfgs(x_best, deadline)
        if polish_result2 is not None:
            pos_pol2, radii_pol2, score_pol2 = polish_result2
            if score_pol2 > best_score:
                best_score = score_pol2
                best_pos = pos_pol2.copy()
                best_radii = radii_pol2.copy()
    
    # Final validation and post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Fallback if invalid
    if not _is_valid(best_pos, best_radii):
         pos = np.zeros((N, 2))
         radii = np.zeros(N)
         k = 0
         for i in range(5):
             for j in range(5):
                 if k < N-1:
                     pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                     radii[k] = 0.095
                     k += 1
         pos[k] = [0.5, 0.02]
         radii[k] = 0.01
         best_pos, best_radii = pos, radii
         best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
