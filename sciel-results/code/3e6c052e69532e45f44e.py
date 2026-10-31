"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START

def _penalty(pos, radii):
    """Compute total penalty for boundary and overlap violations."""
    # Boundary violations
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                       np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = min_dist_sq + 1e-8
    overlap_penalty = np.sum((violation**2) / denom)
    
    return boundary_penalty + overlap_penalty

def _objective(x, penalty_weight=500.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    total_penalty = _penalty(pos, radii)
    return -np.sum(radii) + penalty_weight * total_penalty

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check for final result."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii):
    """
    Shrink radii slightly to ensure strict validity.
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
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.99999
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
        
        # Sample 4000 random points
        for _ in range(4000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
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
            best_x, best_y, best_r = 0.5, 0.5, 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_lattice():
    """
    Generate an initial packing based on a lattice structure.
    This provides a different topology than the greedy best-fit.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid, but we only need 25 circles. Add one more.
    # Actually, let's do a 4x4 grid of larger circles and 10 smaller ones?
    # Or just a 5x5 grid where we pick the best 26 positions?
    # Let's try a 5x5 grid of equal circles first, then optimize.
    
    # Grid spacing
    # If we have 5x5=25 circles, spacing is roughly 1/5 = 0.2
    # Radius would be 0.1
    # Let's place them at (0.1, 0.1), (0.3, 0.1), ...
    
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < N:
                pos[idx] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                radii[idx] = 0.1
                idx += 1
    
    # The 26th circle? 
    # Wait, 5x5 is 25. We need 26.
    # Let's add a 26th circle somewhere in the middle or corner?
    # Actually, a 5x5 grid leaves little room for a 26th.
    # Maybe a 4x4 grid of larger circles + 10 smaller?
    # Let's stick to the 5x5 grid for the first 25, and put the 26th in a gap?
    # Or just randomize the 26th position.
    
    # Let's try a different lattice:
    # 3 rows of 9? No.
    # Let's just use the 5x5 grid for 25, and place the 26th at a random location
    # that maximizes clearance, or just perturb the grid.
    
    # Better: Use a 5x5 grid but slightly smaller circles to fit one more?
    # Or just use the 5x5 grid and let the optimizer figure out the 26th.
    # I'll place the 26th circle at (0.5, 0.5) with a small radius, and let it move.
    # But (0.5, 0.5) is occupied by the center circle of the 5x5 grid.
    
    # Let's shift the grid slightly or use a different arrangement.
    # How about 4 rows of 6? 4*6=24. Plus 2?
    
    # Let's just return the 5x5 grid for 25 circles, and a random 26th.
    if idx < N:
        pos[idx] = [0.5, 0.5]
        radii[idx] = 0.05 # Small radius, will be optimized
    
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed_anneal(pos, radii, n_steps=3000, t_start=0.02, t_end=0.0001, seed=42):
    """
    Force-directed simulated annealing for circle packing.
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
                forces[i, 0] += (r - x) * 50.0
            if x > 1.0 - r:
                forces[i, 0] -= (x - (1.0 - r)) * 50.0
            if y < r:
                forces[i, 1] += (r - y) * 50.0
            if y > 1.0 - r:
                forces[i, 1] -= (y - (1.0 - r)) * 50.0
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r and dist > 1e-6:
                    # Repulsive force
                    overlap = sum_r - dist
                    # Force magnitude proportional to overlap
                    f_mag = overlap * 10.0
                    f_x = (dx / dist) * f_mag
                    f_y = (dy / dist) * f_mag
                    
                    forces[i, 0] += f_x
                    forces[i, 1] += f_y
                    forces[j, 0] -= f_x
                    forces[j, 1] -= f_y
                elif dist <= 1e-6:
                    # If centers coincide, push apart randomly
                    angle = rng.uniform(0, 2*np.pi)
                    f_mag = sum_r * 10.0
                    f_x = np.cos(angle) * f_mag
                    f_y = np.sin(angle) * f_mag
                    forces[i, 0] += f_x
                    forces[i, 1] += f_y
                    forces[j, 0] -= f_x
                    forces[j, 1] -= f_y
        
        # Update positions
        step_size = temp * 0.1
        for i in range(N):
            pos[i, 0] += forces[i, 0] * step_size
            pos[i, 1] += forces[i, 1] * step_size
            
            # Keep within bounds (soft)
            pos[i, 0] = np.clip(pos[i, 0], 0.0, 1.0)
            pos[i, 1] = np.clip(pos[i, 1], 0.0, 1.0)
            
            # Update radius: try to grow if space allows
            # Max radius allowed by walls
            max_r_wall = min(pos[i, 0], 1.0-pos[i, 0], pos[i, 1], 1.0-pos[i, 1])
            
            # Max radius allowed by other circles
            max_r_circ = max_r_wall
            for j in range(N):
                if i == j: continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            # Grow or shrink towards max_r_circ
            target_r = max_r_circ * 0.99
            if target_r > 0:
                radii[i] = radii[i] * 0.9 + target_r * 0.1
                radii[i] = max(radii[i], 1e-5)
            else:
                radii[i] = max(radii[i] * 0.9, 1e-5)
        
        # Ensure validity after update
        pos, radii = _post_process(pos, radii)
        
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Lattice
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    # 3. Random perturbations
    for i in range(3):
        pos_p, rad_p = _initial_random(200 + i)
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
        
    # Optimization stages
    penalty_stages = [100.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Stage 1: L-BFGS-B
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(penalty_stages[0],),
                              options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
            x0 = result.x
        except Exception:
            pass
        
        # Stage 2: L-BFGS-B
        if time.time() < deadline - 30:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[1],),
                                  options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
                x0 = result.x
            except Exception:
                pass
        
        # Stage 3: L-BFGS-B
        if time.time() < deadline - 40:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
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
        
        # Force-directed annealing on the current best
        if time.time() < deadline - 50:
            pos_fa, radii_fa = _force_directed_anneal(best_pos, best_radii, n_steps=2000, seed=idx+1)
            score_fa = np.sum(radii_fa)
            if _is_valid(pos_fa, radii_fa) and score_fa > best_score:
                best_score = score_fa
                best_pos = pos_fa.copy()
                best_radii = radii_fa.copy()
                
                # Polish with L-BFGS-B
                if time.time() < deadline - 60:
                    x0_fa = np.concatenate([best_pos.flatten(), best_radii])
                    try:
                        result = minimize(_objective, x0_fa, method='L-BFGS-B', bounds=bounds,
                                          args=(penalty_stages[2],),
                                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
                        x_opt_fa = result.x
                        pos_opt_fa = x_opt_fa[:2*N].reshape(N, 2)
                        radii_opt_fa = np.maximum(x_opt_fa[2*N:], 1e-4)
                        
                        pos_pp_fa, radii_pp_fa = _post_process(pos_opt_fa, radii_opt_fa)
                        score_pp_fa = np.sum(radii_pp_fa)
                        
                        if _is_valid(pos_pp_fa, radii_pp_fa) and score_pp_fa > best_score:
                            best_score = score_pp_fa
                            best_pos = pos_pp_fa.copy()
                            best_radii = radii_pp_fa.copy()
                    except Exception:
                        pass

    # Final high-precision polish
    if time.time() < deadline - 10:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
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
