"""Circle packing in unit square - 26 circles maximizing sum of radii using SOCP formulation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _objective_slsqp(x, penalty_weight=1.0):
    """
    Objective for SLSQP: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations (soft)
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations using Euclidean distance (SOCP-style)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist - dist)
    overlap_pen = np.sum(viol * viol)
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _constraints_slsqp(x):
    """
    Inequality constraints for SLSQP.
    Each constraint must be >= 0.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    constraints = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N + i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i]) - x[2*N + i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N + i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i+1]) - x[2*N + i]})
    
    # Overlap constraints: ||p_i - p_j||_2 >= r_i + r_j
    # Equivalent to: ||p_i - p_j||_2 - r_i - r_j >= 0
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2) - x[2*N+i] - x[2*N+j]
            })
    
    return constraints

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist[mask] < (min_dist[mask] - tol)):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_grid():
    """Grid-based initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Use a 5x5 grid with some adjustments
    k = 0
    for i in range(5):
        for j in range(5):
            if k >= N:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[k] = 0.095
            k += 1
    
    # Add one more circle at bottom
    if k < N:
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        k += 1
        
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_corner():
    """Corner-biased initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.18
    
    for i in range(4, N):
        # Place remaining circles along edges
        edge = i % 4
        t = (i - 4) / (N - 4) * 0.8 + 0.1
        if edge == 0:
            x, y = t, 0.08
        elif edge == 1:
            x, y = t, 0.92
        elif edge == 2:
            x, y = 0.08, t
        else:
            x, y = 0.92, t
        
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
        
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _solve_slsqp(x0, deadline):
    """Solve using SLSQP with SOCP-style constraints."""
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    constraints = _constraints_slsqp(x0)
    
    try:
        result = minimize(_objective_slsqp, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          args=(1.0,),
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.success or result.fun < _objective_slsqp(x0, 1.0) - 1e-10:
            return result.x
    except Exception:
        pass
        
    return x0

def _solve_penalty(x0, deadline):
    """Solve using penalty method with L-BFGS-B."""
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    current_x = x0.copy()
    
    for p_w in [10.0, 100.0, 1000.0]:
        if time.time() > deadline - 5:
            break
        try:
            result = minimize(_objective_slsqp, current_x, method='L-BFGS-B', bounds=bounds,
                              args=(p_w,),
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < _objective_slsqp(current_x, p_w) - 1e-10:
                current_x = result.x
        except Exception:
            pass
        
    return current_x

def _polish(pos, radii, deadline):
    """High-precision polishing."""
    x = np.concatenate([pos.flatten(), radii])
    
    # Try SLSQP polish
    if time.time() < deadline - 5:
        x = _solve_slsqp(x, deadline)
    
    # Try penalty polish
    if time.time() < deadline - 5:
        x = _solve_penalty(x, deadline)
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_grid, rad_grid = _initial_grid()
    inits.append((pos_grid, rad_grid))
    
    pos_corner, rad_corner = _initial_corner()
    inits.append((pos_corner, rad_corner))
    
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i * 10)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_grid.copy()
    best_radii = rad_grid.copy()
    best_score = np.sum(best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: SLSQP with constraints
        if time.time() < deadline - 10:
            current_x = _solve_slsqp(current_x, deadline)
            
        # Stage 2: Penalty method
        if time.time() < deadline - 5:
            current_x = _solve_penalty(current_x, deadline)
            
        pos_opt = current_x[:2*N].reshape(N, 2)
        radii_opt = np.maximum(current_x[2*N:], 1e-4)
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to grid
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
