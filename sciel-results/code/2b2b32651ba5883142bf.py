"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

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
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
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

def _objective_slsqp(x):
    """Objective for SLSQP: minimize -sum(radii)."""
    radii = x[2*N:]
    return -np.sum(radii)

def _boundary_constraints(x, i):
    """Boundary constraints for circle i: x-r>=0, 1-x-r>=0, y-r>=0, 1-y-r>=0."""
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    return np.array([
        pos[i, 0] - radii[i],
        1.0 - pos[i, 0] - radii[i],
        pos[i, 1] - radii[i],
        1.0 - pos[i, 1] - radii[i]
    ])

def _overlap_constraints(x, i, j):
    """Overlap constraint for circles i and j: dist - (ri+rj) >= 0."""
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    dx = pos[i, 0] - pos[j, 0]
    dy = pos[i, 1] - pos[j, 1]
    dist = np.sqrt(dx*dx + dy*dy)
    return dist - radii[i] - radii[j]

def _slsqp_pack(x0, deadline, max_iter=200):
    """
    Optimize circle packing using SLSQP with explicit constraints.
    x0: initial positions and radii.
    """
    if time.time() > deadline:
        return x0
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999)) # x
        bounds.append((0.001, 0.999)) # y
    for i in range(N):
        bounds.append((0.001, 0.5))   # r
    
    constraints = []
    # Boundary constraints for each circle
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: _boundary_constraints(x, i)
        })
    
    # Overlap constraints for each pair
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: _overlap_constraints(x, i, j)
            })
    
    try:
        result = minimize(
            _objective_slsqp, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        if result.success or result.fun < _objective_slsqp(x0):
            return result.x
    except Exception:
        pass
    return x0

def _initial_grid():
    """Grid initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                radii[k] = 0.09
                k += 1
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate diverse initializations
    inits = []
    pos_grid, rad_grid = _initial_grid()
    inits.append((pos_grid, rad_grid))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos, best_radii = inits[0]
    best_score = np.sum(best_radii)
    
    # Try each initialization with SLSQP
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Run SLSQP
        x_opt = _slsqp_pack(x0, deadline, max_iter=300)
        
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x_opt[2*N:], 1e-4)
        
        # Post-process to ensure validity
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
    # Final polish: Run SLSQP again on the best result found
    if time.time() < deadline - 10:
        x_best = np.concatenate([best_pos.flatten(), best_radii])
        x_polished = _slsqp_pack(x_best, deadline, max_iter=200)
        pos_pol = x_polished[:2*N].reshape(N, 2)
        rad_pol = np.maximum(x_polished[2*N:], 1e-4)
        pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to grid
        pos, radii = _initial_grid()
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
