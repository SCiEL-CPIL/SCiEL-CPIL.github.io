"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    n = len(radii)
    if np.any(radii < -tol):
        return False
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Check distance >= sum of radii
    # dist_sq >= min_dist^2
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity and fix boundary violations."""
    pos = pos.copy()
    radii = radii.copy()
    
    # First, clip positions to be inside [0,1]
    pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
    pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
    
    # Shrink radii to fit boundaries
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        if radii[i] > max_r:
            radii[i] = max_r * 0.99999

    # Iteratively shrink radii to resolve overlaps
    for _ in range(100):
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Shrink both radii slightly
                    # We want dist >= sum_r * (1+epsilon)
                    # Let's just shrink the larger one or both proportionally
                    if dist > 1e-9:
                        scale = dist / sum_r * 0.9999
                    else:
                        scale = 0.5
                    radii[i] *= scale
                    radii[j] *= scale
                    valid = False
        if valid:
            break
            
    return pos, radii

def _initial_corner_biased(seed=7):
    """
    Robust corner-biased initialization.
    Places large circles in corners first, then fills the rest.
    """
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 1. Place 4 large circles in corners
    corner_pos = [
        (0.12, 0.12),
        (0.88, 0.12),
        (0.12, 0.88),
        (0.88, 0.88)
    ]
    for i, (cx, cy) in enumerate(corner_pos):
        pos[i] = [cx, cy]
        radii[i] = 0.12 # Start with a guess, will be adjusted

    # Adjust corner radii to be as large as possible without overlapping each other
    # For now, just set them to a reasonable size
    for i in range(4):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        radii[i] = max_r * 0.95

    # 2. Place remaining circles greedily
    for i in range(4, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        
        # Try many random points
        for _ in range(2000):
            # Bias towards center or edges? Center usually has more space left
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
            
            if max_r > best_r:
                best_r = max_r
                best_x = x
                best_y = y
        
        if best_r < 0.01:
            # Fallback: try to find a spot with small radius
            best_r = 0.01
            best_x = rng.uniform(0.1, 0.9)
            best_y = rng.uniform(0.1, 0.9)
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.95 # Add small margin
        
        # Ensure it doesn't overlap immediately (post-process this single addition)
        # This is expensive if done fully, so we rely on the main optimization
        # But let's do a quick check
        for j in range(i):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j]:
                 # Shrink
                 scale = dist / (radii[i] + radii[j]) * 0.95
                 radii[i] *= scale
                 # Note: this might make it smaller than best_r, which is fine

    return pos, radii

def _slsqp_constraints(x):
    """
    Returns an array of constraint values.
    Constraint is satisfied if value >= 0.
    x = [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    """
    pos = x[0::3] # This is wrong, x is flat [x1, y1, r1, x2, y2, r2...]
    # Let's reshape properly
    # x has 3*N elements
    # x[0], x[1], x[2] -> circle 0
    # x[3], x[4], x[5] -> circle 1
    
    # We need to extract pos and radii
    # pos = x.reshape(-1, 3)[:, :2]
    # radii = x.reshape(-1, 3)[:, 2]
    
    pts = x.reshape(N, 3)
    pos = pts[:, :2]
    radii = pts[:, 2]
    
    constraints = []
    
    # 1. Boundary constraints
    # x_i - r_i >= 0
    constraints.extend(pos[:, 0] - radii)
    # 1 - x_i - r_i >= 0
    constraints.extend(1.0 - pos[:, 0] - radii)
    # y_i - r_i >= 0
    constraints.extend(pos[:, 1] - radii)
    # 1 - y_i - r_i >= 0
    constraints.extend(1.0 - pos[:, 1] - radii)
    
    # 2. Overlap constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            constraints.append(dist_sq - sum_r * sum_r)
            
    return np.array(constraints)

def _objective_slsqp(x):
    """
    Objective to minimize: -sum(radii)
    x = [x1, y1, r1, ...]
    """
    radii = x[2::3] # Wait, x is [x1, y1, r1, x2, y2, r2]
    # So r1 is at index 2, r2 at 5, etc.
    # x[2], x[5], x[8]...
    # x[2::3] works if x is 1D and ordered x, y, r, x, y, r
    return -np.sum(x[2::3])

def _optimize_slsqp(pos, radii, deadline):
    """
    Optimize using SLSQP with hard constraints.
    """
    # Flatten to [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    x0[0::3] = pos[:, 0]
    x0[1::3] = pos[:, 1]
    x0[2::3] = radii
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99)) # x
        bounds.append((0.01, 0.99)) # y
        bounds.append((0.01, 0.5))  # r
    
    constraints = {'type': 'ineq', 'fun': _slsqp_constraints}
    
    try:
        res = minimize(_objective_slsqp, x0, method='SLSQP', 
                       bounds=bounds, 
                       constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
        
        if res.success:
            pos_opt = res.x[0::3].reshape(N)
            pos_opt_y = res.x[1::3].reshape(N)
            radii_opt = res.x[2::3].reshape(N)
            return np.column_stack([pos_opt, pos_opt_y]), radii_opt
        else:
            # Even if not fully successful, if it's valid and better, use it
            pos_opt = res.x[0::3].reshape(N)
            pos_opt_y = res.x[1::3].reshape(N)
            radii_opt = res.x[2::3].reshape(N)
            pos_arr = np.column_stack([pos_opt, pos_opt_y])
            pos_pp, rad_pp = _post_process(pos_arr, radii_opt)
            if _is_valid(pos_pp, rad_pp):
                return pos_pp, rad_pp
    except Exception:
        pass
        
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    pos, radii = _initial_corner_biased(seed=42)
    pos, radii = _post_process(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(best_radii)
    
    # 2. Optimization Loop
    # We will run SLSQP multiple times with different initializations or restarts
    # to escape local minima.
    
    inits = [(pos, radii)]
    
    # Add a few random restarts
    for seed in [1, 2, 3]:
        rng = np.random.RandomState(seed)
        p_r = rng.uniform(0.1, 0.9, (N, 2))
        r_r = rng.uniform(0.05, 0.15, N)
        p_r, r_r = _post_process(p_r, r_r)
        inits.append((p_r, r_r))
        
    for idx, (p_init, r_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        p_opt, r_opt = _optimize_slsqp(p_init, r_init, deadline)
        
        if _is_valid(p_opt, r_opt):
            score = np.sum(r_opt)
            if score > best_score:
                best_score = score
                best_pos = p_opt.copy()
                best_radii = r_opt.copy()
                
        # Refine with another SLSQP run starting from the result
        if time.time() < deadline - 10:
            p_opt2, r_opt2 = _optimize_slsqp(p_opt, r_opt, deadline)
            if _is_valid(p_opt2, r_opt2):
                score2 = np.sum(r_opt2)
                if score2 > best_score:
                    best_score = score2
                    best_pos = p_opt2.copy()
                    best_radii = r_opt2.copy()

    # 3. Final Polish: Try to increase radii further
    # We can try to move circles slightly to create space for larger radii.
    # This is effectively what SLSQP does, but we can force it by perturbing positions.
    
    if time.time() < deadline - 5:
        # Perturb positions slightly and re-optimize
        rng = np.random.RandomState(99)
        p_pert = best_pos + rng.normal(0, 0.01, (N, 2))
        p_pert, r_pert = _post_process(p_pert, best_radii)
        
        p_opt3, r_opt3 = _optimize_slsqp(p_pert, r_pert, deadline)
        if _is_valid(p_opt3, r_opt3):
            score3 = np.sum(r_opt3)
            if score3 > best_score:
                best_score = score3
                best_pos = p_opt3.copy()
                best_radii = r_opt3.copy()

    # Final validation and fallback
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to a simple grid
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = _post_process(pos, radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
