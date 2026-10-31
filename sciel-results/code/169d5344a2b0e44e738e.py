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
    
    # 2. Pairwise Overlap Violations with soft-min
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii_safe[:, np.newaxis] + radii_safe[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Violation in squared distance space
    violation = min_dist_sq - dist_sq  # positive means overlap
    
    # Soft-min: approximate min of violations with log-sum-exp
    # For overlapping pairs, use smooth approximation
    # g(v) = -beta * log(sum(exp(-beta * v))) approximates min(v) as beta->inf
    # We want to penalize positive violations, so use:
    # penalty = soft_max(violation, beta) for the overlapping pairs
    
    # Only consider upper triangle
    v = violation[mask]
    
    # Soft-max approximation: beta * log(sum(exp(v/beta)))
    # Clip to avoid overflow
    v_clipped = np.clip(v, -50*beta, 50*beta)
    max_v = np.max(v_clipped)
    soft_max = max_v + beta * np.log(np.sum(np.exp((v_clipped - max_v) / beta)))
    
    # Only penalize if there's any positive violation
    overlap_penalty = np.maximum(0.0, soft_max) ** 2
    
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
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale both radii to make them just touch
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _vectorized_refine(pos, radii, n_iter=100, step_size=0.01):
    """
    Vectorized gradient-free refinement using coordinate descent on positions.
    For each circle, try moving it in small steps to increase its radius.
    """
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    
    for it in range(n_iter):
        improved = False
        
        for i in range(N):
            # Try moving circle i in 8 directions
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_pos = pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    
                    # Check bounds
                    if new_pos[i, 0] < radii[i] or new_pos[i, 0] > 1.0 - radii[i]:
                        continue
                    if new_pos[i, 1] < radii[i] or new_pos[i, 1] > 1.0 - radii[i]:
                        continue
                    
                    # Check overlaps with other circles
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        ddx = new_pos[i, 0] - pos[j, 0]
                        ddy = new_pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(ddx*ddx + ddy*ddy)
                        if dist < radii[i] + radii[j]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Try to increase radius of circle i
                    max_r = min(new_pos[i, 0], 1.0-new_pos[i, 0], 
                                new_pos[i, 1], 1.0-new_pos[i, 1])
                    for j in range(N):
                        if i == j:
                            continue
                        ddx = new_pos[i, 0] - pos[j, 0]
                        ddy = new_pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(ddx*ddx + ddy*ddy)
                        max_r = min(max_r, dist - radii[j])
                    
                    if max_r > radii[i] * 1.001:
                        pos = new_pos
                        radii[i] = max_r * 0.9999
                        improved = True
                        break
        
        if not improved:
            break
    
    return pos, radii

def _high_precision_polish(pos, radii, time_limit=10.0):
    """
    High-precision polish using SLSQP with tight tolerances.
    """
    start_time = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-6, 0.5))
    
    # Define constraints for SLSQP
    def _boundary_constraints(x):
        pos_c = x[:2*N].reshape(N, 2)
        radii_c = x[2*N:]
        # Each circle must be inside [0,1]^2
        cons = []
        for i in range(N):
            cons.append(pos_c[i, 0] - radii_c[i])
            cons.append(1.0 - pos_c[i, 0] - radii_c[i])
            cons.append(pos_c[i, 1] - radii_c[i])
            cons.append(1.0 - pos_c[i, 1] - radii_c[i])
        return cons
    
    def _overlap_constraints(x):
        pos_c = x[:2*N].reshape(N, 2)
        radii_c = x[2*N:]
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dx = pos_c[i, 0] - pos_c[j, 0]
                dy = pos_c[i, 1] - pos_c[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii_c[i] + radii_c[j]
                cons.append(dist_sq - min_dist*min_dist)
        return cons
    
    constraints = [{'type': 'ineq', 'fun': _boundary_constraints},
                   {'type': 'ineq', 'fun': _overlap_constraints}]
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    try:
        result = minimize(lambda x: -np.sum(x[2*N:]), x0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-12})
        if result.success:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-6)
            
            # Post-process
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
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
        
        # Stage 2b: Smooth objective for better gradients near boundaries
        if time.time() < deadline - 22:
            try:
                result = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[1], 50.0),
                                  options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
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
            
            # Vectorized refinement
            if time.time() < deadline - 32:
                pos_opt, radii_opt = _vectorized_refine(pos_opt, radii_opt, n_iter=50, step_size=0.005)
            
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

    # Final high-precision polish on the best result
    if time.time() < deadline - 10:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            # Voronoi radius update
            if time.time() < deadline - 15:
                pos_opt, radii_opt = _voronoi_radius_update(pos_opt, radii_opt, n_iter=50)
            
            # Vectorized refinement for final squeeze
            if time.time() < deadline - 18:
                pos_opt, radii_opt = _vectorized_refine(pos_opt, radii_opt, n_iter=30, step_size=0.002)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # High-precision SLSQP polish if time permits
    if time.time() < deadline - 12:
        try:
            pos_hp, radii_hp = _high_precision_polish(best_pos, best_radii, time_limit=8.0)
            score = np.sum(radii_hp)
            if _is_valid(pos_hp, radii_hp) and score > best_score:
                best_score = score
                best_pos = pos_hp.copy()
                best_radii = radii_hp.copy()
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
