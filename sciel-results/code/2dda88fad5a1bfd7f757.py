"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Smooth L-BFGS-B Optimization + High-Precision SLSQP Polish.
1. Initialize with multiple diverse strategies (hexagonal, random, corners).
2. Use L-BFGS-B to minimize a smooth objective: -Sum(radii) + Penalty(soft_min(0, dist - sum_r)).
   This avoids the sharp corners of max(0, x) and allows L-BFGS to converge well.
3. Polish the result with SLSQP using hard inequality constraints and tight tolerances.
4. Return the best valid solution found."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import distance

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    # Check for overlap: dist < sum_r
    # We check if dist_sq < (sum_r - tol)^2 to be safe
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii and adjust positions to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(100):
        valid = True
        # Boundary check
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Shrink the smaller one more, or both proportionally
                    # To preserve sum, shrink both slightly
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.10
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
    
    rng = np.random.RandomState(42)
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        min_dist = min(x, 1 - x, y, 1 - y)
        for j in range(k):
            d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
            min_dist = min(min_dist, d - radii[j])
        if min_dist > 0.02:
            pos[k] = [x, y]
            radii[k] = min_dist * 0.9
            k += 1
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _init_random(seed):
    """Random initialization with rejection sampling for small circles."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(N):
        for _ in range(1000):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(i):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[i] = [x, y]
                radii[i] = min_dist * 0.8
                break
        else:
            # If failed, place in a corner with tiny radius
            pos[i] = [0.05, 0.05]
            radii[i] = 0.01
    return pos, radii

def _softplus(x, beta=100.0):
    """Numerically stable softplus function."""
    # log(1 + exp(x)) approximated
    # For large x, it's x. For small x, it's log(2).
    # We want a smooth approximation of max(0, x).
    # max(0, x) = softplus(x) - log(2)
    # So we use softplus(x) - log(2) as the penalty term.
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def _objective(x, beta=100.0, lambda_pen=1000.0):
    """
    Objective function for L-BFGS-B.
    x = [x0, y0, ..., x25, y25, r0, ..., r25]
    Minimize: -Sum(r) + lambda * Sum( max(0, r_i + r_j - dist_ij)^2 ) + lambda * Boundary_Violation
    Using softplus to smooth the max(0, .) terms.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Objective: -Sum(radii)
    obj = -np.sum(radii)
    
    # Overlap Penalty
    # We want to penalize when dist < r_i + r_j
    # Let v_ij = r_i + r_j - dist_ij
    # Penalty = max(0, v_ij)^2
    # Smooth: (softplus(beta * v_ij) - log(2)/beta)^2 ? No, simpler:
    # Use softplus(beta * v_ij) as a smooth max(0, v_ij)
    # Then square it? Or just sum the softplus terms?
    # Squaring makes the gradient zero when v<0 (approximately), which is good.
    # Let's use: sum( (softplus(beta * v_ij) - log(2)/beta) ** 2 )
    # Actually, just summing softplus(beta * v_ij) is a common smooth max(0, v).
    # Let's try: lambda * sum( softplus(beta * (r_i + r_j - dist_ij)) )
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(dist_sq + 1e-10)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    v = min_dist - dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Smooth max(0, v)
    smooth_max = _softplus(beta * v[mask])
    obj += lambda_pen * np.sum(smooth_max)
    
    # Boundary Penalty
    # x - r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    # Violations: r - x, r - (1-x), r - y, r - (1-y)
    viol_x = radii - pos[:, 0]
    viol_1x = radii - (1.0 - pos[:, 0])
    viol_y = radii - pos[:, 1]
    viol_1y = radii - (1.0 - pos[:, 1])
    
    obj += lambda_pen * np.sum(_softplus(beta * viol_x))
    obj += lambda_pen * np.sum(_softplus(beta * viol_1x))
    obj += lambda_pen * np.sum(_softplus(beta * viol_y))
    obj += lambda_pen * np.sum(_softplus(beta * viol_1y))
    
    return obj

def _lbfgs_optimize(pos, radii, deadline, beta=100.0, lambda_pen=1000.0):
    """Run L-BFGS-B optimization."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.001, 0.5))
    
    def obj_wrapper(x):
        return _objective(x, beta, lambda_pen)
    
    try:
        result = minimize(obj_wrapper, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000})
        
        if time.time() > deadline - 10:
            return pos, radii
            
        x_opt = result.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = x_opt[2*N:]
        
        # Post-process to ensure validity
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
        else:
            return pos, radii
            
    except Exception as e:
        return pos, radii

def _slsqp_polish(pos, radii, deadline):
    """
    High-precision SLSQP polish.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.4))
    
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i, j=j:
                                (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 -
                                (x[2*N+i] + x[2*N+j])**2})
    
    def objective(x):
        return -np.sum(x[2*N:])
    
    try:
        # Use SLSQP with tight tolerances
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-12, 'iprint': 0})
        
        if time.time() > deadline - 5:
            return pos, radii
            
        if result.success or result.fun < objective(x0):
            x = result.x
            pos_out = x[:2*N].reshape(N, 2)
            radii_out = np.maximum(x[2*N:], 1e-4)
            pos_out, radii_out = _post_process(pos_out, radii_out)
            if _is_valid(pos_out, radii_out):
                return pos_out, radii_out
    except Exception:
        pass
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # List of initializations to try
    inits = [
        _init_hexagonal(),
        _init_random(1),
        _init_random(42),
        _init_random(123),
    ]
    
    best_pos = None
    best_radii = None
    best_score = -1.0
    
    for pos, radii in inits:
        if time.time() > deadline - 20:
            break
            
        # Phase 1: L-BFGS-B Optimization
        pos_opt, radii_opt = _lbfgs_optimize(pos, radii, deadline)
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
        # Phase 2: SLSQP Polish
        if time.time() < deadline - 10:
            pos_sl, rad_sl = _slsqp_polish(pos_pp, radii_pp, deadline)
            pos_pp2, rad_pp2 = _post_process(pos_sl, rad_sl)
            if _is_valid(pos_pp2, rad_pp2):
                score2 = np.sum(rad_pp2)
                if score2 > best_score:
                    best_score = score2
                    best_pos = pos_pp2.copy()
                    best_radii = rad_pp2.copy()
    
    # Final Polish on the best found
    if best_pos is not None and time.time() < deadline - 5:
        pos_final, rad_final = _slsqp_polish(best_pos, best_radii, deadline)
        pos_pp, rad_pp = _post_process(pos_final, rad_final)
        if _is_valid(pos_pp, rad_pp):
            score = np.sum(rad_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = rad_pp.copy()
    
    if best_pos is None:
        # Fallback to hexagonal
        pos, radii = _init_hexagonal()
        pos, radii = _post_process(pos, radii)
        best_pos, best_radii = pos, radii
    else:
        # Ensure final validity
        best_pos, best_radii = _post_process(best_pos, best_radii)
        if not _is_valid(best_pos, best_radii):
             pos, radii = _init_hexagonal()
             pos, radii = _post_process(pos, radii)
             best_pos, best_radii = pos, radii

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
