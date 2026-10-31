"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
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
    
    # Check for overlaps: dist_sq >= (r_i + r_j)^2
    # We use a small tolerance to account for floating point errors
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii, iterations=50):
    """
    Shrink radii and adjust positions to ensure strict validity.
    This acts as a repair step if the optimizer leaves slight violations.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(iterations):
        valid = True
        
        # Check walls
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.999, 1e-6)
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale both radii down to fit
                    # This is a simple heuristic repair
                    factor = dist / (sum_r + 1e-12) * 0.999
                    radii[i] = max(radii[i] * factor, 1e-6)
                    radii[j] = max(radii[j] * factor, 1e-6)
                    valid = False
                    
        if valid:
            break
            
    return pos, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r * 0.95  # Slightly smaller to ensure feasibility
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18 * 0.95
        
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        
        if max_r < 0.01:
            for attempt in range(200):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
        
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _objective(x):
    """
    Objective: minimize -sum(radii)
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    radii = x[2*N:]
    return -np.sum(radii)

def _constraints(x):
    """
    Define inequality constraints for SLSQP.
    Constraint: g(x) >= 0
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    cons = []
    
    # Wall constraints:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    for i in range(N):
        cons.append(pos[i, 0] - radii[i])
        cons.append(1.0 - pos[i, 0] - radii[i])
        cons.append(pos[i, 1] - radii[i])
        cons.append(1.0 - pos[i, 1] - radii[i])
        
    # Overlap constraints:
    # dist(i, j)^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            cons.append(dist_sq - sum_r * sum_r)
            
    return cons

def _bounds():
    """
    Define bounds for variables.
    """
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x_i
        bounds.append((0.0, 1.0)) # y_i
    for i in range(N):
        bounds.append((0.0, 0.5)) # r_i
    return bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
        
    best_pos = pos_hex.copy()
    best_radii = rad_hex.copy()
    best_score = np.sum(best_radii)
    
    bounds = _bounds()
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        try:
            # SLSQP with hard constraints
            # We use a moderate maxiter to balance quality and speed
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              constraints=_constraints,
                              options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            
            if result.success or result.fun < -np.sum(rad_init):
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(result.x[2*N:], 1e-6)
                
                # Post-process to ensure strict validity
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
                        
        except Exception as e:
            # If SLSQP fails, try a simpler penalty-based approach as fallback
            pass

    # Final Polish: Use SLSQP again on the best solution found, with tighter tolerances
    if time.time() < deadline - 10:
        try:
            x_best = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective, x_best, method='SLSQP', bounds=bounds,
                              constraints=_constraints,
                              options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False})
            
            if result.success:
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(result.x[2*N:], 1e-6)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
        except Exception:
            pass

    # Final validation and repair
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to a valid grid if all else fails
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
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
