"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

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
    
    # Smoother penalty: (violation)^2 / (min_dist^2 + epsilon)
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = min_dist_sq + 1e-8
    smooth_overlap = (violation ** 2) / denom
    
    overlap_penalty = np.sum(smooth_overlap[mask])
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + penalty_weight * total_penalty

def _objective_smooth(x, penalty_weight=500.0, beta=50.0):
    """
    Smooth objective using soft-min (log-sum-exp) for overlap penalty.
    This provides better gradients near the constraint boundary.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are positive
    radii_safe = np.maximum(radii, 1e-6)
    
    # 1. Boundary Violations (same as before)
    left_viol = np.maximum(0.0, radii_safe - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii_safe))
    bottom_viol = np.maximum(0.0, radii_safe - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii_safe))
    
    boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                       np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # 2. Pairwise Overlap using soft-max for smooth penalty
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii_safe[:, np.newaxis] + radii_safe[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # violation = min_dist_sq - dist_sq (positive when overlapping)
    violation = min_dist_sq - dist_sq
    
    # Soft penalty: use log-sum-exp to create a smooth surrogate
    # For each pair, penalty = softmax(violation * beta) * (1/beta)
    # This is 0 when no violation, grows smoothly when overlapping
    v_masked = np.where(mask, violation, -np.inf)
    # log-sum-exp trick for numerical stability
    v_max = np.max(v_masked, axis=1, keepdims=True)
    # Only compute for pairs that exist (mask is symmetric so use full matrix)
    # Actually, let's compute per-pair smooth penalty
    # For each pair (i,j) with i<j:
    # smooth_viol = log(1 + exp(beta * violation_ij)) / beta
    # This is ~0 when violation < 0, ~violation when violation >> 0
    
    # Compute element-wise
    beta_v = beta * violation
    # log(1 + exp(x)) = max(0,x) + log(1+exp(-|x|)) for stability
    beta_v_clipped = np.clip(beta_v, -500, 500)
    log_sum_exp = np.maximum(0, beta_v_clipped) + np.log1p(np.exp(-np.abs(beta_v_clipped)))
    smooth_viol_per_pair = log_sum_exp / beta
    
    # Only sum over upper triangle
    smooth_overlap = np.sum(smooth_viol_per_pair[mask])
    
    # Scale to be comparable to squared penalty
    overlap_penalty = smooth_overlap * 2.0
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii_safe) + penalty_weight * total_penalty

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
    Shrink radii to ensure strict validity.
    Strategy: shrink only the smaller circle on overlap to preserve larger ones.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(300):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        # 2. Overlap fix: shrink the smaller circle preferentially
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-9:
                    # Shrink to make them just touch with safety margin
                    target_dist = dist * 0.99995
                    if target_dist <= 0:
                        # Circles are essentially at same point, shrink both
                        k = 0.99
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                    else:
                        # Shrink the smaller circle more
                        r_min = min(radii[i], radii[j])
                        r_max = max(radii[i], radii[j])
                        # We need r_i + r_j <= target_dist
                        # Keep r_max mostly intact, shrink r_min
                        max_allowed_small = target_dist - r_max
                        if max_allowed_small < 1e-5:
                            # Both need to shrink
                            total_needed = r_max - target_dist
                            # Split proportionally
                            frac_i = radii[i] / sum_r
                            frac_j = radii[j] / sum_r
                            shrink = sum_r - target_dist
                            radii[i] = max(radii[i] - shrink * frac_i, 1e-5)
                            radii[j] = max(radii[j] - shrink * frac_j, 1e-5)
                        else:
                            if radii[i] <= radii[j]:
                                radii[i] = max(max_allowed_small, 1e-5)
                            else:
                                radii[j] = max(max_allowed_small, 1e-5)
                    valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample 3000 random points for better coverage
        for _ in range(3000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
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
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_lattice():
    """
    Generate a hexagonal lattice-like initial state.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        if row == 5:
            x = 0.5
            y = row * dy + r
            if k < N:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
            continue
            
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        placed = False
        for _ in range(100):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            min_dist = min(x, 1-x, y, 1-y)
            for j in range(k):
                d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                placed = True
                break
        if not placed:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _voronoi_radius_update(pos, radii, n_iter=50):
    """
    Update radii based on Voronoi cell distances.
    For each circle, compute the minimum distance to other circles and walls,
    then set radius to half of that distance (with a safety factor).
    This is a simple iterative refinement that can improve packing density.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(n_iter):
        for i in range(N):
            # Distance to walls
            min_dist_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            
            # Distance to other circles
            min_dist_circ = min_dist_wall
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # The maximum radius for circle i given circle j's position and radius
                # dist >= r_i + r_j  =>  r_i <= dist - r_j
                available = dist - radii[j]
                if available < min_dist_circ:
                    min_dist_circ = available
            
            # New radius is half the minimum distance (so circles just touch)
            new_r = min_dist_circ * 0.9999
            # Don't let radius decrease too much in one step
            if new_r > radii[i]:
                radii[i] = new_r
            else:
                # Allow slight decrease for better packing
                radii[i] = max(new_r, radii[i] * 0.95)
        
        # Post-process to ensure validity
        pos, radii = _post_process(pos, radii)
        
    return pos, radii

def _force_directed_anneal(pos, radii, steps=2000, start_temp=0.05, end_temp=0.001, seed=42):
    """
    Force-directed annealing to escape local optima.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for step in range(steps):
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Calculate forces
        forces = np.zeros_like(pos)
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-12
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r:
                    # Overlap: repulsive force
                    overlap = sum_r - dist
                    force_mag = overlap * 10.0
                    fx = (dx / dist) * force_mag
                    fy = (dy / dist) * force_mag
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]
        
        # Attractive force towards center of available space (weak)
        for i in range(N):
            # Pull towards center if near boundary
            cx = 0.5 - pos[i, 0]
            cy = 0.5 - pos[i, 1]
            forces[i] += [cx * 0.01, cy * 0.01]
        
        # Apply forces with annealing
        for i in range(N):
            pos[i, 0] += forces[i, 0] * temp + rng.normal(0, temp * 0.1)
            pos[i, 1] += forces[i, 1] * temp + rng.normal(0, temp * 0.1)
            
            # Keep within bounds
            pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
            pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])
        
        # Update radii using Voronoi-like approach
        for i in range(N):
            min_dist_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            min_dist_circ = min_dist_wall
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                available = dist - radii[j]
                if available < min_dist_circ:
                    min_dist_circ = available
            
            new_r = min_dist_circ * 0.9999
            # Smooth update
            radii[i] = 0.7 * radii[i] + 0.3 * new_r
        
        # Post-process
        pos, radii = _post_process(pos, radii)
        
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Strategy: Multi-start with different initializations
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Hexagonal Lattice
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    # 3. Perturbed Best Fit (4 variations for more diversity)
    for i in range(4):
        pos_p, rad_p = _initial_random(100 + i)
        inits.append((pos_p, rad_p))
        
    # 4. One more random with different seed
    pos_r, rad_r = _initial_random(999)
    inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    # Multi-stage optimization with increasing penalty weight
    penalty_stages = [100.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Stage 1: Coarse optimization with low penalty
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(penalty_stages[0],),
                              options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-6})
            x0 = result.x
        except Exception:
            pass
        
        # Stage 2: Medium penalty
        if time.time() < deadline - 20:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[1],),
                                  options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
                x0 = result.x
            except Exception:
                pass
        
        # Stage 3: High penalty for final polish
        if time.time() < deadline - 25:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
                x0 = result.x
            except Exception:
                pass
            
            x_opt = x0
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            # Voronoi radius update
            if time.time() < deadline - 30:
                pos_opt, radii_opt = _voronoi_radius_update(pos_opt, radii_opt, n_iter=30)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # If this was the first init, do a Powell polish on the best result
        if idx == 0 and time.time() < deadline - 30:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            try:
                result = minimize(_objective, x0, method='Powell', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 800, 'ftol': 1e-14})
                if result.fun < _objective(x0, penalty_stages[2]) - 1e-12:
                    x_opt = result.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                    
                    # Voronoi radius update
                    if time.time() < deadline - 35:
                        pos_opt, radii_opt = _voronoi_radius_update(pos_opt, radii_opt, n_iter=30)
                    
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass

    # Final high-precision polish using SMOOTH objective for better gradients
    if time.time() < deadline - 10:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            # Use smooth objective with moderate beta for better conditioning
            result = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                              args=(2000.0, 30.0),
                              options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            # Voronoi radius update
            if time.time() < deadline - 15:
                pos_opt, radii_opt = _voronoi_radius_update(pos_opt, radii_opt, n_iter=50)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # Additional polish pass with higher beta for sharper constraints
    if time.time() < deadline - 8:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0, 80.0),
                              options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-13})
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

    # Force-directed anneal on the best result if time permits
    if time.time() < deadline - 20:
        try:
            pos_fa, radii_fa = _force_directed_anneal(best_pos, best_radii, steps=1500, seed=123)
            score = np.sum(radii_fa)
            if _is_valid(pos_fa, radii_fa) and score > best_score:
                best_score = score
                best_pos = pos_fa.copy()
                best_radii = radii_fa.copy()
        except Exception:
            pass

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
