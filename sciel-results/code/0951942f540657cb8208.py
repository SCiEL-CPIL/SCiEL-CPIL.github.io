"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _check_validity(pos, radii, tol=1e-9):
    """
    Check if the packing is valid (disjoint interiors, inside unit square).
    Returns True if valid.
    """
    # Boundary check
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    return True

def _get_max_radius(pos, radii, i):
    """
    Calculate the maximum possible radius for circle i, given current positions of others.
    """
    x, y = pos[i]
    max_r = min(x, 1.0 - x, y, 1.0 - y)
    for j in range(N):
        if i == j:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, dist - radii[j])
    return max(0.0, max_r)

def _greedy_init(seed=0):
    """
    Greedy initialization: place circles one by one, choosing the largest possible radius
    at a random location, or optimizing location for max radius.
    """
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # First circle: center
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_r = 0.0
        best_x, best_y = 0.5, 0.5
        
        # Try many random positions to find a good spot
        for _ in range(2000):
            x = rng.uniform(0.01, 0.99)
            y = rng.uniform(0.01, 0.99)
            
            # Max radius at this position
            r_max = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                r_max = min(r_max, dist - radii[j])
                
            if r_max > best_r:
                best_r = r_max
                best_x = x
                best_y = y
                
        # Add a small safety margin to ensure strict validity
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.999
        
        # If we couldn't find a good spot, shrink existing circles slightly to make room?
        # For now, just accept small radius.
        if best_r < 0.001:
            pos[i] = [best_x, best_y]
            radii[i] = 0.001
            
    return pos, radii

def _simulated_annealing_feasible(pos_init, radii_init, t0=0.1, t_end=0.001, n_steps=20000, seed=0, deadline=None):
    """
    Simulated Annealing that only explores feasible states.
    Moves:
    1. Grow a random circle by delta_r. If infeasible, reject.
    2. Move a random circle by (dx, dy). If infeasible, reject.
    3. Move a circle to a random nearby position. If infeasible, reject.
    
    Objective: Maximize sum(radii).
    """
    rng = np.random.RandomState(seed)
    pos = pos_init.copy().astype(float)
    radii = radii_init.copy().astype(float)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    current_score = best_score
    
    # Check initial validity
    if not _check_validity(pos, radii):
        # Force validity by shrinking
        for i in range(N):
            radii[i] = _get_max_radius(pos, radii, i) * 0.99
        current_score = np.sum(radii)
        best_score = current_score
        best_pos = pos.copy()
        best_radii = radii.copy()

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        # Choose move type
        move_type = rng.random()
        
        old_pos = pos.copy()
        old_radii = radii.copy()
        
        if move_type < 0.4:
            # Grow a random circle
            i = rng.randint(N)
            delta_r = rng.uniform(0.001, 0.02)
            new_r = radii[i] + delta_r
            max_r_allowed = _get_max_radius(pos, radii, i)
            if new_r <= max_r_allowed:
                radii[i] = new_r
            else:
                # Try to grow to max allowed
                radii[i] = max_r_allowed
                
        elif move_type < 0.8:
            # Move a random circle slightly
            i = rng.randint(N)
            dx = rng.normal(0, 0.01)
            dy = rng.normal(0, 0.01)
            pos[i, 0] += dx
            pos[i, 1] += dy
            # Clamp to box
            pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
            pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])
            
            # If move causes overlap, revert
            if not _check_validity(pos, radii):
                pos[i, 0] -= dx
                pos[i, 1] -= dy
                # Re-clamp
                pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
                pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])
                if not _check_validity(pos, radii):
                    # Revert completely
                    pos = old_pos
                    radii = old_radii
                    continue
        else:
            # Swap positions of two circles (if radii are similar, this might be valid)
            # Or just move one circle to the position of another (if that spot is free for its size)
            i = rng.randint(N)
            j = rng.randint(N)
            if i != j:
                # Try to move i to where j is, if i fits there
                # This is complex to check efficiently, so let's just do a random teleport within bounds
                x = rng.uniform(radii[i], 1.0 - radii[i])
                y = rng.uniform(radii[i], 1.0 - radii[i])
                pos[i] = [x, y]
                if not _check_validity(pos, radii):
                    pos = old_pos
                    radii = old_radii
                    continue

        # Evaluate new state
        new_score = np.sum(radii)
        delta_e = new_score - current_score
        
        # Accept if better, or with probability exp(-delta_e / T) if worse
        if delta_e > 0 or (temp > 0 and rng.random() < np.exp(-delta_e / temp)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            # Reject
            pos = old_pos
            radii = old_radii
            
    return best_pos, best_radii

def _local_optimize(pos_init, radii_init, deadline):
    """
    Use SLSQP to maximize sum(radii) subject to non-overlap and boundary constraints.
    """
    x0 = np.concatenate([pos_init.flatten(), radii_init])
    
    # Objective: minimize -sum(radii)
    def obj(x):
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        return -np.sum(radii)
        
    # Constraints: 
    # 1. Boundary: pos[i] - r[i] >= 0, pos[i] + r[i] <= 1
    # 2. Overlap: dist(i,j) >= r[i] + r[j]
    
    # SLSQP requires equality/inequality constraints in form c(x) >= 0
    constraints = []
    
    # Boundary constraints (4 per circle)
    for i in range(N):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        # 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
        
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            # dist(i,j)^2 - (r_i + r_j)^2 >= 0
            # Note: sqrt is non-differentiable at 0, but SLSQP handles it okay usually.
            # Better to use: dist(i,j) - (r_i + r_j) >= 0
            def overlap_con(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                return dist - (x[2*N+i] + x[2*N+j])
            constraints.append({'type': 'ineq', 'fun': overlap_con})
            
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    for i in range(N):
        bounds.append((0.0, 0.5)) # r
        
    try:
        result = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-8})
        if result.success or result.fun < obj(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            
            # Final check and correction
            if _check_validity(pos_opt, radii_opt):
                return pos_opt, radii_opt
            else:
                # Try to recover by shrinking
                for i in range(N):
                    radii_opt[i] = _get_max_radius(pos_opt, radii_opt, i) * 0.999
                if _check_validity(pos_opt, radii_opt):
                    return pos_opt, radii_opt
    except Exception:
        pass
        
    return pos_init, radii_init

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -1.0
    
    # 1. Greedy Initialization
    pos_g, rad_g = _greedy_init(seed=123)
    if _check_validity(pos_g, rad_g):
        score_g = np.sum(rad_g)
        if score_g > best_score:
            best_score = score_g
            best_pos = pos_g
            best_radii = rad_g
            
    # 2. Simulated Annealing (Feasible)
    if time.time() < deadline - 20:
        pos_sa, rad_sa = _simulated_annealing_feasible(
            best_pos, best_radii, 
            t0=0.05, t_end=0.001, 
            n_steps=10000, 
            seed=42, 
            deadline=deadline - 15
        )
        if _check_validity(pos_sa, rad_sa):
            score_sa = np.sum(rad_sa)
            if score_sa > best_score:
                best_score = score_sa
                best_pos = pos_sa
                best_radii = rad_sa
                
    # 3. Local Optimization (SLSQP)
    if time.time() < deadline - 10:
        pos_lo, rad_lo = _local_optimize(best_pos, best_radii, deadline)
        if _check_validity(pos_lo, rad_lo):
            score_lo = np.sum(rad_lo)
            if score_lo > best_score:
                best_score = score_lo
                best_pos = pos_lo
                best_radii = rad_lo
                
    # 4. Another round of SA from the polished result (if time permits)
    if time.time() < deadline - 5:
        pos_sa2, rad_sa2 = _simulated_annealing_feasible(
            best_pos, best_radii, 
            t0=0.02, t_end=0.0005, 
            n_steps=5000, 
            seed=99, 
            deadline=deadline - 2
        )
        if _check_validity(pos_sa2, rad_sa2):
            score_sa2 = np.sum(rad_sa2)
            if score_sa2 > best_score:
                best_score = score_sa2
                best_pos = pos_sa2
                best_radii = rad_sa2

    # Final fallback if something went wrong
    if best_pos is None or not _check_validity(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            pos[i] = [(i % 5) * 0.2 + 0.1, (i // 5) * 0.2 + 0.1]
            radii[i] = 0.09
        best_pos, best_radii = pos, radii
        # Ensure validity
        for i in range(N):
            best_radii[i] = _get_max_radius(best_pos, best_radii, i) * 0.999

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
