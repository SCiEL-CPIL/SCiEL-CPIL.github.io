"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _objective(vars):
    """
    Objective function for L-BFGS-B.
    vars: array of shape (3*N,) containing [x0, y0, r0, x1, y1, r1, ...]
    Returns: -sum(radii) + penalty * (overlap_violations^2 + boundary_violations^2)
    """
    pos = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    # Penalty weight
    penalty_weight = 1000.0
    
    # Boundary violations
    # x - r < 0 => r - x > 0
    # x + r > 1 => x + r - 1 > 0
    bv_left = np.maximum(0, radii - pos[:, 0])
    bv_right = np.maximum(0, pos[:, 0] + radii - 1.0)
    bv_bottom = np.maximum(0, radii - pos[:, 1])
    bv_top = np.maximum(0, pos[:, 1] + radii - 1.0)
    
    boundary_viol = np.sum(bv_left**2) + np.sum(bv_right**2) + np.sum(bv_bottom**2) + np.sum(bv_top**2)
    
    # Overlap violations
    # d_ij < r_i + r_j
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # dist = sqrt(dist_sq)
    # To avoid sqrt in gradient calculation issues, we can use dist_sq?
    # No, violation is linear in dist. 
    # If dist_sq is small, sqrt is fine.
    dist = np.sqrt(dist_sq)
    
    viol = np.maximum(0, min_dist - dist)
    overlap_viol = np.sum(viol[mask] ** 2)
    
    total_viol = boundary_viol + overlap_viol
    score = -np.sum(radii) + penalty_weight * total_viol
    return score

def _is_valid(pos, radii, tol=1e-8):
    """Check if configuration is strictly valid."""
    # Boundary
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    # Overlap
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    dist = np.sqrt(dist_sq)
    if np.any(dist[mask] < min_dist[mask] - tol):
        return False
        
    return True

def _enforce_validity(pos, radii):
    """
    Greedily shrink radii to ensure validity.
    Returns valid pos, radii.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # 1. Boundary constraints
    for i in range(N):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if radii[i] > max_r:
            radii[i] = max(0.0, max_r)
            
    # 2. Overlap constraints
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if sum_r > 1e-10:
                        scale = dist / sum_r
                        radii[i] *= scale
                        radii[j] *= scale
                    else:
                        radii[i] = 0.0
                        radii[j] = 0.0
                    changed = True
        if not changed:
            break
            
    # Final safety clip
    for i in range(N):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if radii[i] > max_r:
            radii[i] = max_r
            
    return pos, radii

def _optimize_single_start(pos_init, radii_init, rng):
    """
    Run L-BFGS-B from a specific initial state.
    """
    # Flatten variables
    vars_init = np.concatenate([pos_init.flatten(), radii_init])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    # L-BFGS-B options
    options = {
        'maxiter': 1000,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxfun': 10000
    }
    
    try:
        result = minimize(_objective, vars_init, method='L-BFGS-B', bounds=bounds, options=options)
        if result.success or result.status == 0:
            vars_opt = result.x
        else:
            # If it fails, maybe use the best point found?
            # L-BFGS-B usually returns a valid point even if it doesn't converge perfectly.
            vars_opt = result.x
            
        pos_opt = vars_opt[:2*N].reshape(N, 2)
        radii_opt = vars_opt[2*N:]
        
        # Enforce validity to be safe
        pos_opt, radii_opt = _enforce_validity(pos_opt, radii_opt)
        
        return pos_opt, radii_opt
        
    except Exception:
        return pos_init, radii_init

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    n_starts = 50 # L-BFGS is fast, so we can do more starts
    time_per_start = (deadline - start_time) / n_starts
    
    for s in range(n_starts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        if s == 0:
            # Grid init
            pos = np.zeros((N, 2))
            radii = np.zeros(N)
            k = 0
            for i in range(5):
                for j in range(5):
                    if k < N:
                        pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                        radii[k] = 0.09
                        k += 1
            if k < N:
                pos[k] = [0.5, 0.5]
                radii[k] = 0.01
        else:
            # Random init
            pos = rng.uniform(0.1, 0.9, (N, 2))
            radii = rng.uniform(0.05, 0.15, N)
            # Enforce validity immediately to get a valid start for L-BFGS
            pos, radii = _enforce_validity(pos, radii)
            
        # Run Optimization
        pos_opt, radii_opt = _optimize_single_start(pos, radii, rng)
        
        # Validate
        if _is_valid(pos_opt, radii_opt):
            score = np.sum(radii_opt)
            if score > best_score:
                best_score = score
                best_pos = pos_opt
                best_radii = radii_opt
                
        # Early stop if we hit a very high score
        if best_score > 2.5:
            break

    # Fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
        best_pos, best_radii = _enforce_validity(pos, radii)

    # Final strict validation and shrink
    best_pos, best_radii = _enforce_validity(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
