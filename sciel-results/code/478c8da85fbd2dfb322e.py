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

def _get_constraints_values(pos, radii):
    """
    Compute all constraint values (must be >= 0 for feasibility).
    Returns array of constraint values.
    """
    n = len(radii)
    constraints = []
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
    constraints.append(pos[:, 0] - radii)  # left
    constraints.append(1.0 - pos[:, 0] - radii)  # right
    constraints.append(pos[:, 1] - radii)  # bottom
    constraints.append(1.0 - pos[:, 1] - radii)  # top
    
    # Overlap constraints: dist - (ri + rj) >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            constraints.append(np.array([dist - radii[i] - radii[j]]))
    
    return np.concatenate(constraints)

def _barrier_objective(x, mu, penalty_weight=0.0):
    """
    Log-barrier augmented objective.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Get constraint values
    cvals = _get_constraints_values(pos, radii)
    
    # Barrier term: -mu * sum(log(max(c, eps)))
    eps = 1e-10
    c_safe = np.maximum(cvals, eps)
    barrier = -mu * np.sum(np.log(c_safe))
    
    # Penalty for any constraint violation (should be small if barrier works)
    viol = np.maximum(0.0, -cvals)
    penalty = penalty_weight * np.sum(viol * viol)
    
    return -np.sum(radii) + barrier + penalty

def _barrier_gradient(x, mu, penalty_weight=0.0):
    """
    Analytical gradient of the barrier objective.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    n = N
    
    grad = np.zeros_like(x)
    
    # Gradient of -sum(radii)
    grad[2*n:] = -1.0
    
    # Gradient of barrier term
    eps = 1e-10
    mu_inv = 1.0 / mu
    
    # Boundary constraints gradients
    # Left: x_i - r_i
    c_left = pos[:, 0] - radii
    c_left_safe = np.maximum(c_left, eps)
    dbarrier_dc = -mu_inv / c_left_safe
    for i in range(n):
        grad[2*i] += dbarrier_dc[i]       # d/dx_i
        grad[2*n+i] += -dbarrier_dc[i]    # d/dr_i
    
    # Right: 1 - x_i - r_i
    c_right = 1.0 - pos[:, 0] - radii
    c_right_safe = np.maximum(c_right, eps)
    dbarrier_dc = -mu_inv / c_right_safe
    for i in range(n):
        grad[2*i] += -dbarrier_dc[i]      # d/dx_i
        grad[2*n+i] += -dbarrier_dc[i]    # d/dr_i
    
    # Bottom: y_i - r_i
    c_bottom = pos[:, 1] - radii
    c_bottom_safe = np.maximum(c_bottom, eps)
    dbarrier_dc = -mu_inv / c_bottom_safe
    for i in range(n):
        grad[2*i+1] += dbarrier_dc[i]     # d/dy_i
        grad[2*n+i] += -dbarrier_dc[i]    # d/dr_i
    
    # Top: 1 - y_i - r_i
    c_top = 1.0 - pos[:, 1] - radii
    c_top_safe = np.maximum(c_top, eps)
    dbarrier_dc = -mu_inv / c_top_safe
    for i in range(n):
        grad[2*i+1] += -dbarrier_dc[i]    # d/dy_i
        grad[2*n+i] += -dbarrier_dc[i]    # d/dr_i
    
    # Overlap constraints gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq + 1e-15)
            
            c = dist - radii[i] - radii[j]
            c_safe = max(c, eps)
            dbarrier_dc = -mu_inv / c_safe
            
            # d(dist)/dx_i = dx/dist, d(dist)/dx_j = -dx/dist
            if dist > 1e-10:
                ddist_dx_i = dx / dist
                ddist_dy_i = dy / dist
            else:
                ddist_dx_i = 0.0
                ddist_dy_i = 0.0
            
            grad[2*i] += dbarrier_dc * ddist_dx_i
            grad[2*i+1] += dbarrier_dc * ddist_dy_i
            grad[2*j] += -dbarrier_dc * ddist_dx_i
            grad[2*j+1] += -dbarrier_dc * ddist_dy_i
            grad[2*n+i] += -dbarrier_dc
            grad[2*n+j] += -dbarrier_dc
    
    # Penalty gradient
    if penalty_weight > 0:
        cvals = _get_constraints_values(pos, radii)
        viol = np.maximum(0.0, -cvals)
        dpenalty_dc = -2.0 * penalty_weight * viol
        
        # Boundary penalties
        c_left = pos[:, 0] - radii
        c_right = 1.0 - pos[:, 0] - radii
        c_bottom = pos[:, 1] - radii
        c_top = 1.0 - pos[:, 1] - radii
        
        for i in range(n):
            if c_left[i] < 0:
                grad[2*i] += dpenalty_dc[i]
                grad[2*n+i] += -dpenalty_dc[i]
            if c_right[i] < 0:
                grad[2*i] += -dpenalty_dc[i + n]
                grad[2*n+i] += -dpenalty_dc[i + n]
            if c_bottom[i] < 0:
                grad[2*i+1] += dpenalty_dc[i + 2*n]
                grad[2*n+i] += -dpenalty_dc[i + 2*n]
            if c_top[i] < 0:
                grad[2*i+1] += -dpenalty_dc[i + 3*n]
                grad[2*n+i] += -dpenalty_dc[i + 3*n]
        
        # Overlap penalties
        idx = 4 * n
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq + 1e-15)
                
                c = dist - radii[i] - radii[j]
                if c < 0:
                    dpen = -2.0 * penalty_weight * (-c)
                    if dist > 1e-10:
                        ddist_dx_i = dx / dist
                        ddist_dy_i = dy / dist
                    else:
                        ddist_dx_i = 0.0
                        ddist_dy_i = 0.0
                    
                    grad[2*i] += dpen * ddist_dx_i
                    grad[2*i+1] += dpen * ddist_dy_i
                    grad[2*j] += -dpen * ddist_dx_i
                    grad[2*j+1] += -dpen * ddist_dy_i
                    grad[2*n+i] += -dpen
                    grad[2*n+j] += -dpen
    
    return grad

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

def _try_radius_expansion(pos, radii, max_iter=50, step_size=0.003):
    """
    Try to expand radii of circles that are not touching any constraint.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            touching_boundary = False
            if abs(pos[i, 0] - radii[i]) < 1e-5 or abs(pos[i, 0] + radii[i] - 1.0) < 1e-5:
                touching_boundary = True
            if abs(pos[i, 1] - radii[i]) < 1e-5 or abs(pos[i, 1] + radii[i] - 1.0) < 1e-5:
                touching_boundary = True
            
            touching_circle = False
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] + 1e-5:
                    touching_circle = True
                    break
            
            if not touching_boundary and not touching_circle:
                max_expansion = 0.01
                max_expansion = min(max_expansion, pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, pos[i, 1] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 1] - radii[i])
                
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    space = dist - radii[i] - radii[j]
                    max_expansion = min(max_expansion, space)
                
                if max_expansion > 1e-6:
                    expansion = min(step_size, max_expansion * 0.5)
                    radii[i] += expansion
                    improved = True
        
        if not improved:
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

def _slsqp_constraints(x):
    """Returns list of constraint dictionaries for SLSQP."""
    n = len(x) // 3
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    constraints = []
    
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})
        
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2) - (x[2*n+i] + x[2*n+j])
            })
            
    return constraints

def _objective_slsqp(x):
    """Objective for SLSQP: minimize -sum(radii)."""
    n = len(x) // 3
    radii = x[2*n:]
    return -np.sum(radii)

def _interior_point_optimize(x0, bounds, deadline, max_mu_steps=15):
    """
    Custom interior-point optimization using log-barrier.
    Returns optimized x.
    """
    current_x = x0.copy()
    mu = 1.0  # Start with large barrier parameter
    mu_decay = 0.5
    
    for mu_step in range(max_mu_steps):
        if time.time() > deadline - 10:
            break
        
        # Minimize barrier-augmented objective
        try:
            result = minimize(
                _barrier_objective,
                current_x,
                jac=_barrier_gradient,
                args=(mu,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < _barrier_objective(current_x, mu):
                current_x = result.x
        except Exception:
            pass
        
        # Check feasibility
        pos = current_x[:2*N].reshape(N, 2)
        radii = np.maximum(current_x[2*N:], 1e-6)
        if _is_valid(pos, radii, tol=1e-6):
            # Try to reduce mu
            mu *= mu_decay
        else:
            # If infeasible, try to repair
            pos_pp, radii_pp = _post_process(pos, radii)
            current_x = np.concatenate([pos_pp.flatten(), radii_pp])
    
    return current_x

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
        
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Interior-point with log-barrier
        if time.time() < deadline - 20:
            current_x = _interior_point_optimize(current_x, bounds, deadline)
            
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Stage 2: SLSQP with hard constraints
        if time.time() < deadline - 12:
            pos_start = current_x[:2*N].reshape(N, 2)
            radii_start = np.maximum(current_x[2*N:], 1e-4)
            pos_start, radii_start = _post_process(pos_start, radii_start)
            
            try:
                x_slsqp0 = np.concatenate([pos_start.flatten(), radii_start])
                cons = _slsqp_constraints(x_slsqp0)
                
                result = minimize(_objective_slsqp, x_slsqp0, method='SLSQP', bounds=bounds,
                                  constraints=cons,
                                  options={'maxiter': 200, 'ftol': 1e-14})
                
                if result.success or result.fun < _objective_slsqp(x_slsqp0):
                    pos_sl = result.x[:2*N].reshape(N, 2)
                    radii_sl = np.maximum(result.x[2*N:], 1e-4)
                    pos_pp, radii_pp = _post_process(pos_sl, radii_sl)
                    score = np.sum(radii_pp)
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
                        
            except Exception:
                pass
    
    # Final polish: try radius expansion on best result
    if time.time() < deadline - 8:
        pos_exp, radii_exp = _try_radius_expansion(best_pos, best_radii)
        pos_pp, radii_pp = _post_process(pos_exp, radii_exp)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    # Final SLSQP polish on the best result
    if time.time() < deadline - 5:
        try:
            pos_final, radii_final = _post_process(best_pos, best_radii)
            x_final0 = np.concatenate([pos_final.flatten(), radii_final])
            cons_final = _slsqp_constraints(x_final0)
            
            result = minimize(_objective_slsqp, x_final0, method='SLSQP', bounds=bounds,
                              constraints=cons_final,
                              options={'maxiter': 250, 'ftol': 1e-14})
            
            if result.success:
                pos_pf = result.x[:2*N].reshape(N, 2)
                radii_pf = np.maximum(result.x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_pf, radii_pf)
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
