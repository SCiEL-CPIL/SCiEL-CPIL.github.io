"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    """
    Objective with softplus-based smooth penalties.
    Uses a soft-min (log-sum-exp) surrogate for the distance to the boundary
    and a smooth max(0, required - dist) for overlaps, which is more
    differentiable than the squared-penalty objective.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)

    # Smooth boundary: soft-min of the four wall distances
    d = np.stack([pos[:, 0], 1.0 - pos[:, 0], pos[:, 1], 1.0 - pos[:, 1]], axis=1)
    neg_d = -d
    m = np.max(neg_d, axis=1, keepdims=True)
    neg_d_clipped = np.clip(neg_d, -50.0, 50.0)
    soft_max_neg = m + np.log(np.sum(np.exp(neg_d_clipped - m), axis=1, keepdims=True))
    soft_min_d = -soft_max_neg[:, 0]
    boundary_pen = np.sum(_soft_max(radii - soft_min_d, beta))

    # Smooth overlap: for each pair, required = r_i + r_j, dist = ||p_i - p_j||
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol_raw = min_dist - dist
    overlap_pen = np.sum(_soft_max(viol_raw[mask], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

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

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
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

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _incremental_energy(pos, radii, changed_idx, penalty_weight=50.0):
    """
    Fast incremental energy computation for a single changed circle.
    Only recomputes terms involving the changed circle.
    """
    n = len(radii)
    x, y, r = pos[changed_idx, 0], pos[changed_idx, 1], radii[changed_idx]
    
    # Boundary penalty for changed circle
    bv = np.maximum(0.0, r - x) + np.maximum(0.0, x - (1.0 - r)) + \
         np.maximum(0.0, r - y) + np.maximum(0.0, y - (1.0 - r))
    boundary_pen = bv * bv
    
    # Overlap penalty with all other circles
    overlap_pen = 0.0
    for j in range(n):
        if j == changed_idx:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r + radii[j]
        viol = max(0.0, min_dist * min_dist - dist_sq)
        overlap_pen += viol * viol
    
    return penalty_weight * (boundary_pen + overlap_pen)

def _full_energy(pos, radii, penalty_weight=50.0):
    """Full energy computation (same as _objective but returns penalty only)."""
    n = len(radii)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return penalty_weight * (boundary_pen + overlap_pen)

def _simulated_annealing(pos, radii, n_steps=3000, t0=0.05, t_end=0.001, seed=0, deadline=None):
    """
    Simulated annealing with efficient energy evaluation.
    Uses incremental updates for speed.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Precompute base penalty (all pairs except those involving index i)
    # We'll maintain the full penalty and update incrementally
    
    current_penalty = _full_energy(pos, radii, 50.0)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        # Perturb: move one circle slightly
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.02)
        dy = rng.normal(0, temp * 0.02)
        dr = rng.normal(0, temp * 0.01)
        
        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        new_r = radii[i] + dr
        
        # Clamp
        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        new_r = np.clip(new_r, 0.01, 0.4)
        
        # Compute delta penalty efficiently
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        
        # Old contribution of circle i to penalty
        old_bv = np.maximum(0.0, old_r - old_x) + np.maximum(0.0, old_x - (1.0 - old_r)) + \
                 np.maximum(0.0, old_r - old_y) + np.maximum(0.0, old_y - (1.0 - old_r))
        old_bpen = old_bv * old_bv
        
        old_open = 0.0
        for j in range(n):
            if j == i:
                continue
            dx_j = old_x - pos[j, 0]
            dy_j = old_y - pos[j, 1]
            dist_sq_j = dx_j * dx_j + dy_j * dy_j
            min_dist_j = old_r + radii[j]
            viol_j = max(0.0, min_dist_j * min_dist_j - dist_sq_j)
            old_open += viol_j * viol_j
        
        old_contrib = old_bpen + old_open
        
        # New contribution of circle i to penalty
        new_bv = np.maximum(0.0, new_r - new_x) + np.maximum(0.0, new_x - (1.0 - new_r)) + \
                 np.maximum(0.0, new_r - new_y) + np.maximum(0.0, new_y - (1.0 - new_r))
        new_bpen = new_bv * new_bv
        
        new_open = 0.0
        for j in range(n):
            if j == i:
                continue
            dx_j = new_x - pos[j, 0]
            dy_j = new_y - pos[j, 1]
            dist_sq_j = dx_j * dx_j + dy_j * dy_j
            min_dist_j = new_r + radii[j]
            viol_j = max(0.0, min_dist_j * min_dist_j - dist_sq_j)
            new_open += viol_j * viol_j
        
        new_contrib = new_bpen + new_open
        
        delta_e = 50.0 * (new_contrib - old_contrib)
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            # Accept
            pos[i, 0] = new_x
            pos[i, 1] = new_y
            radii[i] = new_r
            current_penalty += delta_e
        # else: reject, keep old values
        
        # Periodically check validity and update best
        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    # Final check
    pos_pp, radii_pp = _post_process(pos, radii)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _slsqp_constraints(pos, radii):
    """Generate constraint functions for SLSQP with hard non-overlap constraints."""
    n = len(radii)
    constraints = []
    
    # Boundary constraints: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*n+i] + x[2*n+j])**2
            })
    
    return constraints

def _polish(pos, radii, deadline):
    """High-precision polishing with SLSQP hard constraints."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    best_x = x.copy()
    best_score = np.sum(np.maximum(x[2*N:], 1e-6))
    
    # SLSQP with hard constraints - Pass 1
    if time.time() < deadline - 15:
        try:
            cons = _slsqp_constraints(pos, radii)
            result = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), x, method='SLSQP',
                              bounds=bounds, constraints=cons,
                              options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < -best_score + 1e-10:
                score = np.sum(np.maximum(result.x[2*N:], 1e-6))
                if score > best_score:
                    best_score = score
                    best_x = result.x
        except Exception:
            pass
    
    # SLSQP with hard constraints - Pass 2 (tighter)
    if time.time() < deadline - 5:
        try:
            pos_curr = best_x[:2*N].reshape(N, 2)
            rad_curr = np.maximum(best_x[2*N:], 1e-6)
            cons = _slsqp_constraints(pos_curr, rad_curr)
            result = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), best_x, method='SLSQP',
                              bounds=bounds, constraints=cons,
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.success or result.fun < -best_score + 1e-12:
                score = np.sum(np.maximum(result.x[2*N:], 1e-6))
                if score > best_score:
                    best_score = score
                    best_x = result.x
        except Exception:
            pass
    
    pos_out = best_x[:2*N].reshape(N, 2)
    radii_out = np.maximum(best_x[2*N:], 1e-4)
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    penalty_stages = [50.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Standard objective
        for p_w in penalty_stages:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Stage 2: Smooth objective
        if time.time() < deadline - 10:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 30.0),
                                      options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.fun < _objective_smooth(current_x, p_w, 30.0) - 1e-10:
                        current_x = result.x
                except Exception:
                    pass
                
                pos_opt = current_x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(current_x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Stage 3: Simulated annealing with more steps
        if time.time() < deadline - 15:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=5000, t0=0.04, t_end=0.0003,
                    seed=idx * 100,
                    deadline=deadline
                )
                pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final polish - multiple attempts from different starting points
    # We spend more time here as SLSQP is more effective for final tightening
    
    # Attempt 1: Direct polish
    if time.time() < deadline - 30:
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
    
    # Attempt 2: Perturbed polish (small jitter)
    if time.time() < deadline - 20:
        try:
            rng = np.random.RandomState(12345)
            perturbed_pos = best_pos + rng.normal(0, 0.001, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.01, 0.99)
            # Keep radii from best to start SLSQP close to feasible
            pos_pol2, rad_pol2 = _polish(perturbed_pos, best_radii.copy(), deadline)
            pos_pp2, radii_pp2 = _post_process(pos_pol2, rad_pol2)
            score2 = np.sum(radii_pp2)
            if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
                best_score = score2
                best_pos = pos_pp2.copy()
                best_radii = radii_pp2.copy()
        except Exception:
            pass
            
    # Attempt 3: Perturbed polish (medium jitter)
    if time.time() < deadline - 10:
        try:
            rng = np.random.RandomState(99999)
            perturbed_pos3 = best_pos + rng.normal(0, 0.005, (N, 2))
            perturbed_pos3 = np.clip(perturbed_pos3, 0.01, 0.99)
            pos_pol3, rad_pol3 = _polish(perturbed_pos3, best_radii.copy(), deadline)
            pos_pp3, radii_pp3 = _post_process(pos_pol3, rad_pol3)
            score3 = np.sum(radii_pp3)
            if _is_valid(pos_pp3, radii_pp3) and score3 > best_score:
                best_score = score3
                best_pos = pos_pp3.copy()
                best_radii = radii_pp3.copy()
        except Exception:
            pass
            
    # Attempt 4: Final tight polish
    if time.time() < deadline - 2:
        try:
            pos_pol4, rad_pol4 = _polish(best_pos, best_radii, deadline)
            pos_pp4, radii_pp4 = _post_process(pos_pol4, rad_pol4)
            score4 = np.sum(radii_pp4)
            if _is_valid(pos_pp4, radii_pp4) and score4 > best_score:
                best_score = score4
                best_pos = pos_pp4.copy()
                best_radii = radii_pp4.copy()
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
