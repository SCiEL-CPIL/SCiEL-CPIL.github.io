"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _slsqp_objective(x):
    """
    Objective function for SLSQP.
    Minimizes -sum(radii).
    x = [x1, y1, r1, ..., xN, yN, rN]
    """
    radii = x[2::3]
    return -np.sum(radii)

def _slsqp_constraints():
    """
    Define inequality constraints for SLSQP.
    SLSQP requires g(x) >= 0.
    """
    cons = []
    
    # Boundary constraints
    # x_i >= r_i  => x_i - r_i >= 0
    # 1 - x_i >= r_i => 1 - x_i - r_i >= 0
    # y_i >= r_i  => y_i - r_i >= 0
    # 1 - y_i >= r_i => 1 - y_i - r_i >= 0
    
    for i in range(N):
        # x_i - r_i >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i] - x[3*i+2]
        })
        # 1 - x_i - r_i >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[3*i] - x[3*i+2]
        })
        # y_i - r_i >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]
        })
        # 1 - y_i - r_i >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[3*i+1] - x[3*i+2]
        })
        
    # Overlap constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
            })
            
    return cons

def _is_valid(pos, radii, tol=1e-8):
    """
    Strict validity check.
    """
    if np.any(pos[:, 0] < -tol) or np.any(pos[:, 0] > 1.0 + tol):
        return False
    if np.any(pos[:, 1] < -tol) or np.any(pos[:, 1] > 1.0 + tol):
        return False
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # dist < min_dist => dist_sq < min_dist^2
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _shrink_to_valid(pos, radii, tol=1e-6):
    """
    Shrink radii and adjust positions slightly to ensure validity.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(100):
        valid = True
        
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + tol:
                radii[i] = max(0.0, max_r - tol)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - tol:
                    k = max(0.0, dist / (sum_r + 1e-12))
                    k = min(k, 1.0)
                    radii[i] = radii[i] * k
                    radii[j] = radii[j] * k
                    valid = False
        
        if valid:
            break
            
    return pos, radii

def _run_slsqp(init_x, deadline):
    """
    Run SLSQP optimization.
    """
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-4, 0.45))
        
    cons = _slsqp_constraints()
    
    try:
        result = minimize(
            _slsqp_objective, 
            init_x, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        # Check if it converged or if we are within time
        if time.time() > deadline:
            return None
            
        x_opt = result.x
        pos = x_opt[0::3].reshape(N, 2)
        radii = x_opt[2::3]
        
        # Ensure validity
        if _is_valid(pos, radii):
            return pos, radii
        else:
            pos_val, radii_val = _shrink_to_valid(pos, radii)
            if _is_valid(pos_val, radii_val):
                return pos_val, radii_val
            return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Grid Init
    pos_grid = np.zeros((N, 2))
    rad_grid = np.zeros(N)
    spacing = 1.0 / 5.0
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos_grid[idx] = [(i + 0.5) * spacing, (j + 0.5) * spacing]
                rad_grid[idx] = spacing * 0.4
    if N > 25:
        pos_grid[25] = [0.5, 0.5]
        rad_grid[25] = 0.05
    inits.append((pos_grid, rad_grid))
    
    # 2. Random Init
    rng = np.random.RandomState(42)
    pos_rand = rng.uniform(0.1, 0.9, (N, 2))
    rad_rand = rng.uniform(0.05, 0.1, N)
    inits.append((pos_rand, rad_rand))
    
    # 3. Hexagonal Init
    pos_hex = np.zeros((N, 2))
    rad_hex = np.zeros(N)
    r = 0.1
    dx = r * np.sqrt(3) * 1.05
    dy = r * 1.5 * 1.05
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = r + offset + col * dx
            if 0 <= x <= 1 and 0 <= y <= 1:
                pos_hex[k] = [x, y]
                rad_hex[k] = r
                k += 1
        if k >= N:
            break
    while k < N:
        pos_hex[k] = [0.5, 0.5]
        rad_hex[k] = 0.01
        k += 1
    inits.append((pos_hex, rad_hex))
    
    # 4. Corner-focused Init
    pos_corner = np.zeros((N, 2))
    rad_corner = np.zeros(N)
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
    for i, c in enumerate(corners):
        pos_corner[i] = c
        rad_corner[i] = 0.12
    edges = [
        [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5],
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.5, 0.5]
    ]
    for i in range(4, N):
        if i - 4 < len(edges):
            pos_corner[i] = edges[i - 4]
            rad_corner[i] = 0.06
        else:
            pos_corner[i] = rng.uniform(0.2, 0.8, 2)
            rad_corner[i] = 0.05
    inits.append((pos_corner, rad_corner))
    
    # 5. Dense corner cluster init
    pos_dense = np.zeros((N, 2))
    rad_dense = np.zeros(N)
    corner_pos = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for i, cp in enumerate(corner_pos):
        pos_dense[i] = cp
        rad_dense[i] = 0.14
    mid_positions = [
        [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5],
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
        [0.5, 0.5]
    ]
    for i in range(4, N):
        if i - 4 < len(mid_positions):
            pos_dense[i] = mid_positions[i - 4]
            rad_dense[i] = 0.04
        else:
            pos_dense[i] = rng.uniform(0.2, 0.8, 2)
            rad_dense[i] = 0.03
    inits.append((pos_dense, rad_dense))
    
    # Run SLSQP on each init
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Format x for SLSQP: [x1, y1, r1, x2, y2, r2, ...]
        x_init = np.zeros(3 * N)
        for i in range(N):
            x_init[3*i] = pos_init[i, 0]
            x_init[3*i+1] = pos_init[i, 1]
            x_init[3*i+2] = rad_init[i]
            
        result = _run_slsqp(x_init, deadline - 5)
        
        if result is not None:
            pos_sol, rad_sol = result
            score = np.sum(rad_sol)
            if score > best_score:
                best_score = score
                best_pos = pos_sol.copy()
                best_radii = rad_sol.copy()
                
    # Fallback if SLSQP failed
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                if i*5+j < N-1:
                    pos[i*5+j] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[i*5+j] = 0.09
        pos[N-1] = [0.5, 0.02]
        radii[N-1] = 0.01
        best_pos, best_radii = _shrink_to_valid(pos, radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
