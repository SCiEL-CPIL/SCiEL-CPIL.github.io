"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

def _force_directed_anneal(pos, radii, n_steps=2000, t_start=0.05, t_end=0.001, seed=42):
    """
    Force-directed simulated annealing for circle packing.
    Vectorized for speed.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    
    for step in range(n_steps):
        progress = step / max(n_steps - 1, 1)
        temp = t_start * (t_end / t_start) ** progress
        
        forces = np.zeros_like(pos)
        
        # Wall forces
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            if x < r:
                forces[i, 0] += (r - x) * 10.0
            if x > 1.0 - r:
                forces[i, 0] -= (x - (1.0 - r)) * 10.0
            if y < r:
                forces[i, 1] += (r - y) * 10.0
            if y > 1.0 - r:
                forces[i, 1] -= (y - (1.0 - r)) * 10.0
        
        # Pairwise repulsion (vectorized)
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq + 1e-12)
        
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap force
        overlap = np.maximum(0, min_dist - dist)
        force_mag = overlap * 5.0
        safe_dist = np.where(dist > 1e-10, dist, 1.0)
        fx = (dx / safe_dist) * force_mag
        fy = (dy / safe_dist) * force_mag
        
        # Near-overlap force
        near = np.maximum(0, min_dist * 2.0 - dist) * (dist > min_dist)
        near_force = near * 0.5
        fx_near = (dx / safe_dist) * near_force
        fy_near = (dy / safe_dist) * near_force
        
        total_fx = fx + fx_near
        total_fy = fy + fy_near
        
        forces[:, 0] += np.sum(total_fx, axis=1)
        forces[:, 1] += np.sum(total_fy, axis=1)
        
        # Add noise for annealing
        noise = rng.normal(0, temp, (N, 2))
        
        # Update positions
        pos += forces * 0.1 + noise
        
        # Keep within bounds
        for i in range(N):
            r = radii[i]
            pos[i, 0] = np.clip(pos[i, 0], r, 1.0 - r)
            pos[i, 1] = np.clip(pos[i, 1], r, 1.0 - r)
        
        # Occasionally try to increase radii
        if step % 50 == 0 and temp > t_end * 5:
            for i in range(N):
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                for j in range(N):
                    if j != i:
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        max_r = min(max_r, dist - radii[j])
                
                if max_r > radii[i] * 1.05:
                    new_r = min(max_r * 0.999, radii[i] * 1.02)
                    valid = True
                    for j in range(N):
                        if j != i:
                            dx = pos[i, 0] - pos[j, 0]
                            dy = pos[i, 1] - pos[j, 1]
                            dist = np.sqrt(dx * dx + dy * dy)
                            if dist < new_r + radii[j]:
                                valid = False
                                break
                    if valid:
                        radii[i] = new_r
    
    return pos, radii

def _initial_lattice():
    """
    Generate an initial packing based on a hexagonal lattice pattern.
    This provides a structurally diverse starting point.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal packing parameters
    r_base = 0.12
    dx_hex = 2 * r_base
    dy_hex = np.sqrt(3) * r_base
    
    k = 0
    row = 0
    col = 0
    while k < N:
        x = 0.5 + col * dx_hex * 0.95
        y = 0.5 + row * dy_hex * 0.95
        if col % 2 == 1:
            y += dy_hex * 0.5
        
        # Check if within bounds
        if x - r_base >= 0 and x + r_base <= 1 and y - r_base >= 0 and y + r_base <= 1:
            pos[k] = [x, y]
            radii[k] = r_base
            k += 1
        
        col += 1
        if col > 5:
            col = 0
            row += 1
            if row > 5:
                # Fallback: place remaining circles randomly
                for rem in range(k, N):
                    pos[rem] = [0.5 + random.uniform(-0.3, 0.3), 0.5 + random.uniform(-0.3, 0.3)]
                    radii[rem] = 0.08
                break
    
    # Post-process to ensure validity
    pos, radii = _post_process(pos, radii)
    return pos, radii

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
    
    # Use a smoother penalty with soft-min approach
    # violation = max(0, min_dist - dist) in squared distance terms
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    # Smooth approximation: use log-sum-exp style soft penalty
    # This creates a smooth surrogate that approximates the nonsmooth max
    epsilon = 1e-4
    # Soft penalty: for each pair, penalize based on how much they overlap
    # Using a quadratic penalty that's smooth everywhere
    smooth_overlap = (violation ** 2) / (min_dist_sq + epsilon)
    
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
    
    for _ in range(100):
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

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    This creates a valid, dense configuration that serves as a strong starting point.
    Uses a combination of grid sampling and random sampling for better coverage.
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
                # If we place circle i at (x,y) with radius r, and circle j has radius r_j:
                # dist >= r + r_j  =>  r <= dist - r_j
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            # Fallback: place tiny circle at center
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97  # Slight shrink to ensure no overlap in initial state
        
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    # Force validity by shrinking
    pos, radii = _post_process(pos, radii)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    inits = []
    
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    for i in range(4):
        pos_p, rad_p = _initial_random(100 + i)
        inits.append((pos_p, rad_p))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
    
    penalty_stages = [100.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Stage 1: Coarse
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(penalty_stages[0],),
                              options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-6})
            x0 = result.x
        except Exception:
            pass
        
        # Stage 2: Medium
        if time.time() < deadline - 25:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[1],),
                                  options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
                x0 = result.x
            except Exception:
                pass
        
        # Stage 3: High penalty
        if time.time() < deadline - 30:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
                x0 = result.x
            except Exception:
                pass
            
            x_opt = x0
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Force-directed annealing
        if time.time() < deadline - 35 and idx <= 3:
            try:
                pos_fd, radii_fd = _force_directed_anneal(pos_init, rad_init, 
                                                           n_steps=2000, 
                                                           t_start=0.03, 
                                                           t_end=0.0005,
                                                           seed=1000 + idx)
                pos_fd_pp, radii_fd_pp = _post_process(pos_fd, radii_fd)
                score_fd = np.sum(radii_fd_pp)
                
                if _is_valid(pos_fd_pp, radii_fd_pp) and score_fd > best_score:
                    best_score = score_fd
                    best_pos = pos_fd_pp.copy()
                    best_radii = radii_fd_pp.copy()
            except Exception:
                pass
        
        # Powell polish on first init
        if idx == 0 and time.time() < deadline - 40:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            try:
                result = minimize(_objective, x0, method='Powell', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 1000, 'ftol': 1e-14})
                if result.fun < _objective(x0, penalty_stages[2]) - 1e-12:
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

    # Final high-precision polish
    if time.time() < deadline - 15:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
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
        
        # Force-directed polish
        if time.time() < deadline - 8:
            try:
                pos_fd2, radii_fd2 = _force_directed_anneal(best_pos, best_radii,
                                                             n_steps=1500,
                                                             t_start=0.01,
                                                             t_end=0.0001,
                                                             seed=9999)
                pos_fd2_pp, radii_fd2_pp = _post_process(pos_fd2, radii_fd2)
                score_fd2 = np.sum(radii_fd2_pp)
                
                if _is_valid(pos_fd2_pp, radii_fd2_pp) and score_fd2 > best_score:
                    best_score = score_fd2
                    best_pos = pos_fd2_pp.copy()
                    best_radii = radii_fd2_pp.copy()
            except Exception:
                pass

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
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
