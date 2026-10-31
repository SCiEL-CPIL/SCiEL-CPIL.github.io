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

def _post_process(pos, radii, iters=200):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(iters):
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

def _init_greedy_best_fit():
    """
    Greedy initialization: place circles one by one, choosing the position 
    that allows the largest possible radius.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large circle in the center
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    rng = np.random.RandomState(42)
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        
        # Sample many candidate positions
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            # Max radius allowed by boundaries
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            
            # Max radius allowed by existing circles
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            
            if max_r > best_r:
                best_r = max_r
                best_x = x
                best_y = y
        
        # If no good spot found, try a smaller radius near center
        if best_r < 0.005:
            best_x, best_y = 0.5, 0.5
            best_r = 0.005
            
        pos[i] = [best_x, best_y]
        # Slightly shrink to ensure strict non-overlap for initialization
        radii[i] = best_r * 0.98
        
    return pos, radii

def _init_hexagonal():
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

def _energy(pos, radii, beta=100.0):
    """
    Smooth energy function for Simulated Annealing.
    Minimizes -sum(radii) + penalty * (violations).
    """
    n = len(radii)
    sum_r = np.sum(radii)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist - dist)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -sum_r + beta * (boundary_pen + overlap_pen)

def _simulated_annealing(pos, radii, n_steps=3000, t0=0.05, t_end=0.0001, seed=0):
    """
    Simulated Annealing to explore the configuration space.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    best_energy = _energy(pos, radii)
    current_energy = best_energy
    
    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        # Choose a random circle to perturb
        i = rng.randint(n)
        
        # Perturbation magnitude decreases with temperature
        scale = 0.02 * (1.0 - frac * 0.5)
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        dr = rng.normal(0, scale * 0.5)
        
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        
        new_x = np.clip(pos[i, 0] + dx, 0.01, 0.99)
        new_y = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        new_r = np.clip(radii[i] + dr, 0.01, 0.45)
        
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        new_energy = _energy(pos, radii)
        delta_e = new_energy - current_energy
        
        # Metropolis acceptance criterion
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-10)):
            current_energy = new_energy
        else:
            # Revert
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
            current_energy = best_energy # Reset to last known good? No, keep current valid state
            
        # Track best valid state
        if current_energy < best_energy:
            best_energy = current_energy
            # Check if this state is strictly valid or close enough
            # For SA, we accept slightly invalid states to escape local minima,
            # but we track the best *valid* state separately for final output.
            # However, _energy is smooth, so low energy implies low violation.
            # We will post-process later.
            best_pos = pos.copy()
            best_radii = radii.copy()
            best_score = np.sum(radii)
            
    return best_pos, best_radii

def _slsqp_polish(pos, radii):
    """
    Polish the solution using SLSQP with hard constraints.
    """
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.45))
        
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i]) - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i+1]) - x[2*N+i]})
    
    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            def _non_overlap(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                return dist - x[2*N+i] - x[2*N+j]
            constraints.append({'type': 'ineq', 'fun': _non_overlap})
    
    def _objective(x):
        radii = np.maximum(x[2*N:], 1e-6)
        return -np.sum(radii)
    
    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.success:
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-5)
    return pos_out, radii_out

def _radius_growth_pass(pos, radii):
    """
    Fix positions and maximize radii via SLSQP.
    """
    # Variables are just radii
    x = radii.copy()
    
    bounds = [(1e-5, 0.45) for _ in range(N)]
    
    constraints = []
    
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i, max_r=max_r: max_r - x[i]})
        
    # Non-overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j, dist=dist: dist - x[i] - x[j]})
    
    def _objective(x):
        return -np.sum(x)
    
    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.success:
            x = result.x
    except Exception:
        pass
    
    radii_out = np.maximum(x, 1e-5)
    return radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Try multiple initializations and SA seeds
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
            
        # Alternate between greedy and hexagonal init
        if seed % 2 == 0:
            pos_init, rad_init = _init_greedy_best_fit()
        else:
            pos_init, rad_init = _init_hexagonal()
            
        # 2. Simulated Annealing
        pos_sa, rad_sa = _simulated_annealing(
            pos_init, rad_init, 
            n_steps=2000, 
            t0=0.04, 
            t_end=0.0001, 
            seed=seed
        )
        
        # 3. Post-process to ensure validity
        pos_pp, rad_pp = _post_process(pos_sa, rad_sa, iters=100)
        
        if _is_valid(pos_pp, rad_pp):
            score = np.sum(rad_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = rad_pp.copy()
                
        # 4. SLSQP Polish on the best valid state found so far
        if time.time() < deadline - 10 and best_pos is not None:
            pos_pol, rad_pol = _slsqp_polish(best_pos, best_radii)
            pos_pp2, rad_pp2 = _post_process(pos_pol, rad_pol, iters=200)
            
            if _is_valid(pos_pp2, rad_pp2):
                score2 = np.sum(rad_pp2)
                if score2 > best_score:
                    best_score = score2
                    best_pos = pos_pp2.copy()
                    best_radii = rad_pp2.copy()
                    
            # 5. Radius growth pass
            if time.time() < deadline - 5:
                rad_grown = _radius_growth_pass(best_pos, best_radii)
                # Check validity of grown radii
                if _is_valid(best_pos, rad_grown):
                    score3 = np.sum(rad_grown)
                    if score3 > best_score:
                        best_score = score3
                        best_radii = rad_grown.copy()
                        
    # Final fallback if no valid solution was found (should not happen)
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            pos[i] = [0.5, 0.5]
            radii[i] = 0.01
        best_pos, best_radii = _post_process(pos, radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
