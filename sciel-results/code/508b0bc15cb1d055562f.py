"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
import random
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
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(50):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    This creates a valid, dense configuration that serves as a strong starting point.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        # Try to find a position for the next circle
        # We will try to maximize the radius of the new circle at a given position
        # by checking distance to all existing circles and walls.
        
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        # Randomly sample many positions to find a good spot
        # Use a grid + random jitter for coverage
        rng = np.random.RandomState(i)
        
        # Sample 1000 random points
        for _ in range(1000):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            
            # Max radius allowed by existing circles
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # If we place circle i at (x,y) with radius r, and circle j has radius r_j:
                # dist >= r + r_j  => r <= dist - r_j
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.01:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.01:
            # Fallback: place in a corner if we run out of space?
            # For 26 circles, we should always find space if we start with 0.25.
            # If we really can't, place tiny circle at center (should not happen)
            best_x, best_y = 0.5, 0.5
            best_r = 0.01
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.99 # Slight shrink to ensure no overlap in initial state
        
    return pos, radii

def _initial_lattice():
    """Generate a lattice-based initial configuration."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid with some circles
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                radii[k] = 0.095
                k += 1
    
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.1, N)
    # Force validity by shrinking
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed_anneal(pos, radii, n_steps=2000, dt=0.001, cooling=0.999, time_limit=None, start_time=None):
    """
    Force-directed annealing: push overlapping circles apart, push boundary-violating
    circles inward, and grow radii when possible.
    """
    if start_time is None:
        start_time = time.time()
    
    pos = pos.copy()
    radii = radii.copy()
    temp = 1.0
    
    for step in range(n_steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # 1. Boundary forces: push circles inward if they violate boundaries
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            
            # Left wall
            if x < r:
                forces[i, 0] += (r - x) / (r + 1e-12)
            # Right wall
            if x > 1.0 - r:
                forces[i, 0] -= (x - (1.0 - r)) / (r + 1e-12)
            # Bottom wall
            if y < r:
                forces[i, 1] += (r - y) / (r + 1e-12)
            # Top wall
            if y > 1.0 - r:
                forces[i, 1] -= (y - (1.0 - r)) / (r + 1e-12)
        
        # 2. Pairwise repulsion: push overlapping circles apart
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-12
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r:
                    # Overlap: repulsive force
                    overlap = (sum_r - dist) / sum_r
                    force_mag = overlap * 2.0
                    fx = (dx / dist) * force_mag
                    fy = (dy / dist) * force_mag
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # 3. Grow radii: try to increase each radius slightly
        for i in range(N):
            # Check if we can grow this circle
            can_grow = True
            max_grow = 0.01  # Max growth per step
            
            # Boundary check
            x, y = pos[i]
            r = radii[i]
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            if r >= max_r_wall:
                can_grow = False
            else:
                max_grow = min(max_grow, max_r_wall - r)
            
            # Check against other circles
            if can_grow:
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # If we grow by dr, need dist >= (r+dr) + r_j
                    # dr <= dist - r - r_j
                    possible_dr = dist - r - radii[j]
                    if possible_dr < 0:
                        can_grow = False
                        break
                    else:
                        max_grow = min(max_grow, possible_dr)
            
            if can_grow and max_grow > 0:
                # Grow with some randomness
                grow_amount = min(max_grow, 0.005) * (0.5 + 0.5 * np.random.random())
                radii[i] += grow_amount
        
        # Apply forces with annealing temperature
        for i in range(N):
            # Random perturbation scaled by temperature
            noise = np.random.normal(0, temp * 0.01, 2)
            pos[i] += forces[i] * dt + noise
            
            # Keep within [0,1] bounds (soft)
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        # Cool down
        temp *= cooling
        
        # Ensure radii stay positive
        radii = np.maximum(radii, 1e-5)
        
        # Periodic validity check and fix
        if step % 100 == 0:
            pos, radii = _post_process(pos, radii)
    
    return pos, radii

def _polish(pos, radii, time_limit=5.0):
    """
    High-precision polish using L-BFGS-B with tight tolerances.
    """
    start_time = time.time()
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))
    
    try:
        # First pass: L-BFGS-B with moderate settings
        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                          args=(500.0,),
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        
        x_opt = result.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x_opt[2*N:], 1e-4)
        
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        
        # Second pass: Powell for final polish if time allows
        if time.time() - start_time < time_limit * 0.6:
            x1 = np.concatenate([pos_pp.flatten(), radii_pp])
            result2 = minimize(_objective, x1, method='Powell', bounds=bounds,
                               args=(500.0,),
                               options={'maxiter': 200, 'ftol': 1e-15})
            
            x_opt2 = result2.x
            pos_opt2 = x_opt2[:2*N].reshape(N, 2)
            radii_opt2 = np.maximum(x_opt2[2*N:], 1e-4)
            
            pos_pp2, radii_pp2 = _post_process(pos_opt2, radii_opt2)
            
            if _is_valid(pos_pp2, radii_pp2) and np.sum(radii_pp2) > np.sum(radii_pp):
                return pos_pp2, radii_pp2
        
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
        
    except Exception:
        pass
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Strategy: Multi-start with diverse initializations + force-directed annealing + polish
    
    inits = []
    
    # 1. Best Fit greedy
    pos_bf, rad_bf = _initial_best_fit()
    inits.append(('best_fit', pos_bf, rad_bf))
    
    # 2. Lattice
    pos_lat, rad_lat = _initial_lattice()
    inits.append(('lattice', pos_lat, rad_lat))
    
    # 3. Random perturbations of best-fit (3 variations)
    for i in range(3):
        pos_p, rad_p = _initial_random(100 + i)
        inits.append(('random_{}'.format(i), pos_p, rad_p))
    
    # 4. One more random with different seed
    pos_r, rad_r = _initial_random(999)
    inits.append(('random_999', pos_r, rad_r))
    
    # 5. Perturbed lattice
    rng = np.random.RandomState(42)
    pos_lat_pert = pos_lat + rng.normal(0, 0.02, (N, 2))
    pos_lat_pert = np.clip(pos_lat_pert, 0.05, 0.95)
    rad_lat_pert = rad_lat * (1 + rng.normal(0, 0.1, N))
    rad_lat_pert = np.clip(rad_lat_pert, 0.03, 0.15)
    pos_lat_pert, rad_lat_pert = _post_process(pos_lat_pert, rad_lat_pert)
    inits.append(('lattice_pert', pos_lat_pert, rad_lat_pert))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    # Phase 1: Force-directed annealing on each initialization
    for idx, (name, pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:  # Leave 30s for polish phase
            break
        
        # Run force-directed annealing with time limit
        remaining = deadline - time.time() - 25  # Leave 25s for polish
        if remaining < 5:
            break
        
        pos_fd, rad_fd = _force_directed_anneal(
            pos_init, rad_init, 
            n_steps=3000, 
            dt=0.002, 
            cooling=0.9995,
            time_limit=min(remaining, 15.0),
            start_time=start_time
        )
        
        # Post-process and check validity
        pos_pp, rad_pp = _post_process(pos_fd, rad_fd)
        score = np.sum(rad_pp)
        
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
        
        # Also try L-BFGS-B on the force-directed result if time permits
        if time.time() < deadline - 20:
            x0 = np.concatenate([pos_pp.flatten(), rad_pp])
            bounds = []
            for i in range(N):
                bounds.append((0.005, 0.995))
                bounds.append((0.005, 0.995))
            for i in range(N):
                bounds.append((0.005, 0.5))
            
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(500.0,),
                                  options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
                
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp2, radii_pp2 = _post_process(pos_opt, radii_opt)
                score2 = np.sum(radii_pp2)
                
                if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
                    best_score = score2
                    best_pos = pos_pp2.copy()
                    best_radii = radii_pp2.copy()
                    
            except Exception:
                pass
    
    # Phase 2: Final high-precision polish on the best result
    if time.time() < deadline - 5:
        pos_polished, radii_polished = _polish(best_pos, best_radii, time_limit=min(deadline - time.time() - 3, 10.0))
        
        if _is_valid(pos_polished, radii_polished) and np.sum(radii_polished) > best_score:
            best_pos = pos_polished
            best_radii = radii_polished
            best_score = np.sum(best_radii)
    
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

