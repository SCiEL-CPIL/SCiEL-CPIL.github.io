"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Smooth soft-plus objective + L-BFGS-B from diverse star initializations + SLSQP polish."""
import numpy as np
import time
from scipy.optimize import minimize

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
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(300):
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

def _sum_radii(pos, radii):
    return np.sum(radii)

def _softplus(x, k=200.0):
    """Numerically stable softplus."""
    return np.where(x > 20.0 / k, x, np.log1p(np.exp(k * x)) / k)

def _objective(x, k=200.0):
    """
    Objective to minimize: -sum(radii) + penalty for overlaps and boundary violations.
    x = [x0,y0,x1,y1,...,x25,y25,r0,r1,...,r25]
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = x[2 * N:]
    
    # Boundary penalties: circle must be inside [0,1]^2
    # x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    boundary_viol = 0.0
    for i in range(N):
        boundary_viol += _softplus(-(pos[i, 0] - radii[i]), k)
        boundary_viol += _softplus(-(1.0 - pos[i, 0] - radii[i]), k)
        boundary_viol += _softplus(-(pos[i, 1] - radii[i]), k)
        boundary_viol += _softplus(-(1.0 - pos[i, 1] - radii[i]), k)
    
    # Overlap penalties: dist_ij - r_i - r_j >= 0
    overlap_viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap_viol += _softplus(-(dist - radii[i] - radii[j]), k)
    
    penalty = 5.0 * (boundary_viol + overlap_viol)
    return -np.sum(radii) + penalty

def _objective_grad(x, k=200.0):
    """Gradient of the objective."""
    pos = x[:2 * N].reshape(N, 2)
    radii = x[2 * N:]
    
    grad = np.zeros_like(x)
    
    # Derivative of softplus: sigmoid(k*x)
    def _sigmoid(t):
        # numerically stable
        return np.where(t > 20.0, 1.0, 1.0 / (1.0 + np.exp(-t)))
    
    # Boundary gradients
    for i in range(N):
        # x_i - r_i
        t = k * (pos[i, 0] - radii[i])
        s = _sigmoid(t)
        # d/dx_i of softplus(-(x_i - r_i)) = -s * k / k = -s
        grad[2 * i] += -5.0 * s
        grad[2 * N + i] += 5.0 * s
        
        # 1 - x_i - r_i
        t = k * (1.0 - pos[i, 0] - radii[i])
        s = _sigmoid(t)
        grad[2 * i] += -5.0 * s
        grad[2 * N + i] += 5.0 * s
        
        # y_i - r_i
        t = k * (pos[i, 1] - radii[i])
        s = _sigmoid(t)
        grad[2 * i + 1] += -5.0 * s
        grad[2 * N + i] += 5.0 * s
        
        # 1 - y_i - r_i
        t = k * (1.0 - pos[i, 1] - radii[i])
        s = _sigmoid(t)
        grad[2 * i + 1] += -5.0 * s
        grad[2 * N + i] += 5.0 * s
    
    # Overlap gradients
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq + 1e-12)
            t = k * (dist - radii[i] - radii[j])
            s = _sigmoid(t)
            # d/dx_i of softplus(-(dist - r_i - r_j)) = -s * (dx/dist)
            # d/dr_i = -s
            # d/dx_j = -s * (-dx/dist) = s * dx/dist
            # d/dr_j = -s
            grad[2 * i] += -5.0 * s * (dx / dist)
            grad[2 * i + 1] += -5.0 * s * (dy / dist)
            grad[2 * N + i] += 5.0 * s
            
            grad[2 * j] += 5.0 * s * (dx / dist)
            grad[2 * j + 1] += 5.0 * s * (dy / dist)
            grad[2 * N + j] += 5.0 * s
    
    # Objective gradient: -sum(radii) -> grad[radii] -= 1
    grad[2 * N:] -= 1.0
    
    return grad

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
        for _ in range(200):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
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

def _init_grid():
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
    if k < N:
        pos[k] = [0.5, 0.5]
        radii[k] = 0.01
        k += 1
    return pos, radii

def _init_corners():
    """Corner-biased initialization."""
    rng = np.random.RandomState(42)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02)]
        radii[i] = 0.17
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
            for attempt in range(300):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1 - x2, y2, 1 - y2)
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

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _lbgfbs_b_optimize(pos, radii, deadline, k=200.0, maxiter=200):
    """
    Optimize using L-BFGS-B on the smooth soft-plus objective.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.4))
    
    try:
        result = minimize(_objective, x0, jac=_objective_grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        if result.fun < _objective(x0):
            x = result.x
            pos_out = x[:2 * N].reshape(N, 2)
            radii_out = np.maximum(x[2 * N:], 1e-4)
            pos_out, radii_out = _post_process(pos_out, radii_out)
            if _is_valid(pos_out, radii_out):
                return pos_out, radii_out
    except Exception:
        pass
    
    return pos, radii

def _slsqp_polish(pos, radii, deadline):
    """
    SLSQP with hard inequality constraints for non-overlap and boundaries.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.4))
    
    constraints = []
    
    for i in range(N):
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i, j=j:
                                (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 -
                                (x[2*N+i] + x[2*N+j])**2})
    
    def objective(x):
        return -np.sum(x[2*N:])
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12})
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
    deadline = start_time + 145.0
    
    # Generate diverse initializations
    inits = []
    pos_hex, rad_hex = _init_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_grid, rad_grid = _init_grid()
    inits.append((pos_grid, rad_grid))
    
    pos_cor, rad_cor = _init_corners()
    inits.append((pos_cor, rad_cor))
    
    for s in [100, 200, 300, 400, 500]:
        pos_r, rad_r = _init_random(s)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_hex.copy()
    best_radii = rad_hex.copy()
    best_score = _sum_radii(best_pos, best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        # Phase 1: L-BFGS-B on smooth objective
        pos_lb, rad_lb = _lbgfbs_b_optimize(pos_init, rad_init, deadline, k=200.0, maxiter=150)
        pos_pp, rad_pp = _post_process(pos_lb, rad_lb)
        score = _sum_radii(pos_pp, rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
        
        # Phase 2: SLSQP polish with hard constraints
        if time.time() < deadline - 10:
            pos_sl, rad_sl = _slsqp_polish(pos_pp, rad_pp, deadline)
            pos_pp2, rad_pp2 = _post_process(pos_sl, rad_sl)
            score2 = _sum_radii(pos_pp2, rad_pp2)
            if _is_valid(pos_pp2, rad_pp2) and score2 > best_score:
                best_score = score2
                best_pos = pos_pp2.copy()
                best_radii = rad_pp2.copy()
    
    # Final: SLSQP on the best solution found
    if time.time() < deadline - 8:
        pos_final, rad_final = _slsqp_polish(best_pos, best_radii, deadline)
        pos_pp, rad_pp = _post_process(pos_final, rad_final)
        score = _sum_radii(pos_pp, rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
