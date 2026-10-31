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
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity. Vectorized for speed."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(50):
        valid = True
        
        # Boundary check
        max_r_wall = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]), 
                                np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        too_big_wall = radii > max_r_wall
        if np.any(too_big_wall):
            radii[too_big_wall] = max_r_wall[too_big_wall] * 0.9999
            valid = False
        
        # Overlap check
        # We need to check all pairs. If overlap, shrink both.
        # Doing this iteratively is safer.
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale down both radii to fit in distance
                    # k = dist / sum_r
                    # But we must also respect walls.
                    # Simplest: shrink both by factor k * 0.9999
                    k = (dist / (sum_r + 1e-12)) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """Greedy best-fit initialization with fine grid for first circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Place first circle in center
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        # Use a grid for the first few circles to ensure good placement
        if i < 5:
            grid_size = 50
            xs = np.linspace(0.05, 0.95, grid_size)
            ys = np.linspace(0.05, 0.95, grid_size)
            for x in xs:
                for y in ys:
                    max_r = min(x, 1.0 - x, y, 1.0 - y)
                    for j in range(i):
                        dx = x - pos[j, 0]
                        dy = y - pos[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        possible_r = dist - radii[j]
                        if possible_r < max_r:
                            max_r = possible_r
                    if max_r > best_r and max_r > 0.003:
                        best_r = max_r
                        best_x = x
                        best_y = y
                        found = True
        else:
            # Random search for remaining circles
            rng = np.random.RandomState(i * 1000 + 42)
            for _ in range(2000):
                x = rng.uniform(0.02, 0.98)
                y = rng.uniform(0.02, 0.98)
                
                max_r = min(x, 1.0 - x, y, 1.0 - y)
                for j in range(i):
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    possible_r = dist - radii[j]
                    if possible_r < max_r:
                        max_r = possible_r
                
                if max_r > best_r and max_r > 0.003:
                    best_r = max_r
                    best_x = x
                    best_y = y
                    found = True
        
        if not found or best_r < 0.003:
            # Fallback: place in a corner or center with small radius
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.98
        
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
                radii[k] = r
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

def _objective_positions(x, pos, radii, penalty_weight=1000.0):
    """
    Objective for optimizing positions only.
    x = [x1, y1, ..., xN, yN]
    Radii are fixed.
    """
    pos_new = x.reshape(N, 2)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos_new[:, 0]) + np.maximum(0.0, pos_new[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos_new[:, 1]) + np.maximum(0.0, pos_new[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos_new[:, 0, np.newaxis] - pos_new[np.newaxis, :, 0]
    dy = pos_new[:, 1, np.newaxis] - pos_new[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return penalty_weight * (boundary_pen + overlap_pen)

def _optimize_positions(pos, radii, deadline):
    """Optimize positions to minimize overlaps and boundary violations."""
    x0 = pos.flatten()
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
        
    try:
        result = minimize(_objective_positions, x0, args=(pos, radii, 1000.0), 
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < _objective_positions(x0, pos, radii, 1000.0) - 1e-10:
            return result.x.reshape(N, 2)
    except Exception:
        pass
    return pos

def _optimize_radii(pos, radii, deadline):
    """
    Optimize radii given fixed positions using SLSQP with hard constraints.
    x = [r1, ..., rN]
    Constraints:
    1. r_i >= 0
    2. r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    3. dist(i,j) >= r_i + r_j
    """
    x0 = radii.copy()
    
    # Bounds: r_i <= min distance to wall
    max_r = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]), 
                       np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    bounds = []
    for i in range(N):
        bounds.append((1e-5, max_r[i]))
        
    constraints = []
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j  =>  dist(i,j) - r_i - r_j >= 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            # Constraint: dist - (r_i + r_j) >= 0
            # lambda r: dist - (r[i] + r[j])
            constraints.append({
                'type': 'ineq',
                'fun': lambda r, i=i, j=j, dist=dist: dist - (r[i] + r[j])
            })
            
    # Objective: maximize sum(radii) => minimize -sum(radii)
    def obj(r):
        return -np.sum(r)
        
    try:
        result = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-12})
        if result.success or result.fun < obj(x0) - 1e-10:
            return result.x
    except Exception:
        pass
    return radii

def _full_polish(pos, radii, deadline):
    """
    Final polish using SLSQP on all variables (positions and radii).
    This is expensive but can find the global local optimum.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((1e-5, 0.5))
        
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
        
    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*N+i] + x[2*N+j])**2
            })
            
    def obj(x):
        return -np.sum(x[2*N:])
        
    try:
        result = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-12})
        if result.success or result.fun < obj(x0) - 1e-10:
            pos_out = result.x[:2*N].reshape(N, 2)
            radii_out = result.x[2*N:]
            return pos_out, radii_out
    except Exception:
        pass
        
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        pos_cur = pos_init.copy()
        rad_cur = rad_init.copy()
        
        # Alternating optimization loop
        # 1. Optimize positions to clear overlaps
        # 2. Optimize radii to grow into available space
        # Repeat a few times
        
        for step in range(5):
            if time.time() > deadline - 10:
                break
                
            # Optimize positions
            pos_cur = _optimize_positions(pos_cur, rad_cur, deadline)
            
            # Optimize radii
            rad_cur = _optimize_radii(pos_cur, rad_cur, deadline)
            
            # Post-process to ensure validity
            pos_pp, rad_pp = _post_process(pos_cur, rad_cur)
            if _is_valid(pos_pp, rad_pp):
                score = np.sum(rad_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = rad_pp.copy()
        
        # Final full polish with SLSQP on all variables
        if time.time() < deadline - 5:
            pos_pol, rad_pol = _full_polish(best_pos, best_radii, deadline)
            pos_pp, rad_pp = _post_process(pos_pol, rad_pol)
            if _is_valid(pos_pp, rad_pp):
                score = np.sum(rad_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = rad_pp.copy()
                    
    # Final validation and fallback
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to a known valid configuration
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
