"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x):
    """Objective: minimize negative sum of radii + smooth penalty for violations.
    
    Uses log-sum-exp for smooth gradients on binding constraints.
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = np.maximum(x[2*n:], 1e-4)
    
    # Boundary violations
    boundary_viol = np.zeros(4*n)
    for i in range(n):
        boundary_viol[4*i]   = max(0.0, radii[i] - pos[i, 0])
        boundary_viol[4*i+1] = max(0.0, pos[i, 0] - (1.0 - radii[i]))
        boundary_viol[4*i+2] = max(0.0, radii[i] - pos[i, 1])
        boundary_viol[4*i+3] = max(0.0, pos[i, 1] - (1.0 - radii[i]))
    
    # Pairwise violations
    m = n*(n-1)//2
    pair_viol = np.zeros(m)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            pair_viol[idx] = max(0.0, min_dist - dist)
            idx += 1
    
    # Log-sum-exp penalty
    tau = 0.01
    def _lse_penalty(v):
        if np.max(v) < 1e-10:
            return 0.0
        v_scaled = v / tau
        v_max = np.max(v_scaled)
        lse = v_max + np.log(np.sum(np.exp(v_scaled - v_max)))
        return lse * tau
    
    boundary_pen = _lse_penalty(boundary_viol)
    pair_pen = _lse_penalty(pair_viol)
    
    return -np.sum(radii) + 50.0 * boundary_pen + 50.0 * pair_pen


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = 0.09
    
    pos[25, 0] = 0.1
    pos[25, 1] = 0.1
    radii[25] = 0.08
    
    return pos, radii


def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.09
    for i in range(5):
        idx = i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1
        radii[idx] = r
    
    dy = r * np.sqrt(3)
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + dy
        radii[idx] = r
    
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 2 * dy
        radii[idx] = r
    
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + 3 * dy
        radii[idx] = r
    
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 4 * dy
        radii[idx] = r
    
    pos[25, 0] = 0.5
    pos[25, 1] = 0.95
    radii[25] = 0.05
    
    return pos, radii


def _initial_corner_heavy():
    """Initial configuration with larger corner and edge circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corner circles
    corner_r = 0.15
    corners = [(corner_r, corner_r), (1-corner_r, corner_r), 
               (corner_r, 1-corner_r), (1-corner_r, 1-corner_r)]
    for i, (x, y) in enumerate(corners):
        pos[i, 0] = x
        pos[i, 1] = y
        radii[i] = corner_r
    
    # 8 edge circles
    edge_r = 0.10
    edges = [
        (0.3, edge_r), (0.7, edge_r),
        (0.3, 1-edge_r), (0.7, 1-edge_r),
        (edge_r, 0.3), (edge_r, 0.7),
        (1-edge_r, 0.3), (1-edge_r, 0.7)
    ]
    for i, (x, y) in enumerate(edges):
        pos[4+i, 0] = x
        pos[4+i, 1] = y
        radii[4+i] = edge_r
    
    # 14 interior circles in a grid
    interior_r = 0.07
    cx, cy = 0.5, 0.5
    spacing = 0.18
    count = 0
    for i in range(-2, 3):
        for j in range(-2, 3):
            if 12 + count >= N:
                break
            px = cx + i * spacing
            py = cy + j * spacing
            if 0.05 <= px <= 0.95 and 0.05 <= py <= 0.95:
                pos[12+count, 0] = px
                pos[12+count, 1] = py
                radii[12+count] = interior_r
                count += 1
            if 12 + count >= N:
                break
    
    # Fill any remaining
    while 12 + count < N:
        pos[12+count, 0] = 0.5
        pos[12+count, 1] = 0.5
        radii[12+count] = 0.04
        count += 1
    
    return pos, radii


def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(500):
        valid = True
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.999, 1e-4)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.999)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.999)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.999)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.999
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _grow_and_push(pos, radii, deadline):
    """Iteratively grow each circle and push neighbors to make room."""
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    n = N
    rng = np.random.RandomState(7)
    
    for outer_iter in range(200):
        if time.time() > deadline - 15:
            break
        
        i = rng.randint(n)
        max_growth = 0.05
        
        max_growth = min(max_growth, pos[i, 0] - radii[i])
        max_growth = min(max_growth, 1.0 - pos[i, 0] - radii[i])
        max_growth = min(max_growth, pos[i, 1] - radii[i])
        max_growth = min(max_growth, 1.0 - pos[i, 1] - radii[i])
        
        if max_growth < 1e-6:
            continue
        
        growth = max_growth * 0.5
        new_radii = radii.copy()
        new_radii[i] += growth
        new_pos = pos.copy()
        
        for j in range(n):
            if j == i:
                continue
            dx = new_pos[i, 0] - new_pos[j, 0]
            dy = new_pos[i, 1] - new_pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = new_radii[i] + new_radii[j]
            
            if dist < min_dist - 1e-9:
                if dist < 1e-10:
                    angle = rng.uniform(0, 2 * np.pi)
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                    dist = 1.0
                else:
                    dx /= dist
                    dy /= dist
                
                push = (min_dist - dist) * 0.5
                new_pos[j, 0] += dx * push
                new_pos[j, 1] += dy * push
        
        valid = True
        for k in range(n):
            if new_pos[k, 0] < new_radii[k] or new_pos[k, 0] > 1.0 - new_radii[k]:
                valid = False
                break
            if new_pos[k, 1] < new_radii[k] or new_pos[k, 1] > 1.0 - new_radii[k]:
                valid = False
                break
            if valid:
                for l in range(k + 1, n):
                    dx = new_pos[k, 0] - new_pos[l, 0]
                    dy = new_pos[k, 1] - new_pos[l, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < new_radii[k] + new_radii[l] - 1e-9:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            new_score = np.sum(new_radii)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
                pos = new_pos
                radii = new_radii
    
    return best_pos, best_radii


def _constraints_func(x):
    """Constraint function for trust-constr: all constraints >= 0."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append(x_c[i] - r[i])
        constraints.append(1 - x_c[i] - r[i])
        constraints.append(y_c[i] - r[i])
        constraints.append(1 - y_c[i] - r[i])
        constraints.append(r[i] - 0.001)
    
    # Pairwise distance constraints (using squared distances for efficiency)
    for i in range(N):
        for j in range(i+1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            constraints.append(dist_sq - min_dist*min_dist)
    
    return np.array(constraints)


def _jacobian_constraints(x):
    """Jacobian of all constraints for trust-constr."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    n_constraints = 5*N + N*(N-1)//2
    jac = np.zeros((n_constraints, 3*N))
    
    idx = 0
    for i in range(N):
        # x_i - r_i >= 0
        jac[idx, i] = 1.0
        jac[idx, 2*N+i] = -1.0
        idx += 1
        # 1 - x_i - r_i >= 0
        jac[idx, i] = -1.0
        jac[idx, 2*N+i] = -1.0
        idx += 1
        # y_i - r_i >= 0
        jac[idx, N+i] = 1.0
        jac[idx, 2*N+i] = -1.0
        idx += 1
        # 1 - y_i - r_i >= 0
        jac[idx, N+i] = -1.0
        jac[idx, 2*N+i] = -1.0
        idx += 1
        # r_i - 0.001 >= 0
        jac[idx, 2*N+i] = 1.0
        idx += 1
    
    for i in range(N):
        for j in range(i+1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            min_dist = r[i] + r[j]
            
            jac[idx, i] = 2*dx
            jac[idx, j] = -2*dx
            jac[idx, N+i] = 2*dy
            jac[idx, N+j] = -2*dy
            jac[idx, 2*N+i] = -2*min_dist
            jac[idx, 2*N+j] = -2*min_dist
            idx += 1
    
    return jac


def _gradient_objective(x):
    """Gradient of objective: -sum(radii)."""
    grad = np.zeros(3*N)
    grad[2*N:] = -1.0
    return grad


def _objective_simple(x):
    """Simple objective: negative sum of radii."""
    return -np.sum(x[2*N:])


def _trust_constr_solve(x0, deadline, maxiter=300):
    """Run trust-constr optimization with explicit constraints."""
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
    for i in range(N):
        bounds.append((0, 1))
    for i in range(N):
        bounds.append((0.001, 0.5))
    
    constraint = NonlinearConstraint(
        _constraints_func,
        0, np.inf,
        jac=_jacobian_constraints
    )
    
    try:
        res = minimize(
            _objective_simple,
            x0,
            method='trust-constr',
            jac=_gradient_objective,
            bounds=bounds,
            constraints=[constraint],
            options={'maxiter': maxiter, 'gtol': 1e-8, 'xtol': 1e-10}
        )
        
        if res.fun < _objective_simple(x0):
            # Verify feasibility
            cons_vals = _constraints_func(res.x)
            if cons_vals.min() > -1e-6:
                x_opt = res.x
                pos_opt = np.column_stack([x_opt[:N], x_opt[N:2*N]])
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                return pos_opt, radii_opt
    except Exception:
        pass
    
    return None, None


def _polish(pos, radii, deadline):
    """High-precision local search using trust-constr and L-BFGS-B."""
    x0 = np.concatenate([pos.flatten(), radii])
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Pass 1: trust-constr with explicit constraints (most powerful)
    if time.time() < deadline - 15:
        pos_opt, radii_opt = _trust_constr_solve(x0, deadline, maxiter=400)
        if pos_opt is not None:
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                x0 = np.concatenate([best_pos.flatten(), best_radii])
    
    # Pass 2: L-BFGS-B with penalty method
    if time.time() < deadline - 10:
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((1e-5, 0.5))
        
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Pass 3: Final trust-constr polish
    if time.time() < deadline - 5:
        pos_opt, radii_opt = _trust_constr_solve(x0, deadline, maxiter=500)
        if pos_opt is not None:
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii, best_score


def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # L-BFGS-B first (fast gradient-based)
    if time.time() < deadline - 15:
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.25))
        
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-9})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Grow and push: try to expand circles locally
    if time.time() < deadline - 15:
        gp_pos, gp_radii = _grow_and_push(best_pos, best_radii, deadline)
        gp_score = np.sum(gp_radii)
        if gp_score > best_score:
            best_score = gp_score
            best_pos = gp_pos.copy()
            best_radii = gp_radii.copy()
    
    # Trust-constr with explicit constraints (key improvement)
    if time.time() < deadline - 15:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        pos_opt, radii_opt = _trust_constr_solve(x0, deadline, maxiter=350)
        if pos_opt is not None:
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Powell (derivative-free, good for fine-tuning)
    if time.time() < deadline - 15:
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.25))
        
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 800, 'ftol': 1e-13})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 8:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build initial configurations
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_corner_heavy()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456, 789]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed corner-heavy
    pos, radii = _initial_corner_heavy()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Run optimization from each initial config
    time_per_config = min(18.0, (deadline - start_time - 20) / max(1, len(init_configs)))
    
    for idx, (pos, radii) in enumerate(init_configs):
        if time.time() > deadline - 20:
            break
        
        config_deadline = min(deadline - 10, time.time() + time_per_config)
        pos_opt, radii_opt, score = _optimize_from(pos, radii, config_deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Fallback if nothing valid found
    if best_pos is None or best_score < 0:
        pos, radii = _initial_grid()
        best_pos = pos
        best_radii = radii
        best_score = np.sum(radii)
    
    # Final post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Final high-precision polish on the best solution
    if time.time() < deadline - 3:
        best_pos, best_radii, _ = _polish(best_pos, best_radii, deadline)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
