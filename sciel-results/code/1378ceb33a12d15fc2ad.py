"""Circle packing in unit square - 26 circles maximizing sum of radii."""
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
import numpy as np
import time

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


def _post_process(pos, radii, tol=1e-9):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(500):
        valid = True
        
        for i in range(N):
            # Boundary constraints
            max_r_x = min(pos[i, 0], 1.0 - pos[i, 0])
            max_r_y = min(pos[i, 1], 1.0 - pos[i, 1])
            max_r = max_r_x if max_r_x < max_r_y else max_r_y
            if radii[i] > max_r - tol:
                radii[i] = max(max_r - 1e-7, 1e-6)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist - tol:
                    # Scale both radii to just fit
                    scale = dist / (min_dist + 1e-12) * 0.999
                    radii[i] = max(radii[i] * scale, 1e-6)
                    radii[j] = max(radii[j] * scale, 1e-6)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _objective_penalty(x, lam=1000.0):
    """Penalty-based objective for unconstrained optimizers."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = np.maximum(x[2*n:], 1e-6)
    
    penalty = 0.0
    # Boundary penalties
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    penalty += np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Pairwise overlap penalties
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    penalty += np.sum(overlap[mask] ** 2 / (4.0 * np.maximum(min_dist_sq[mask], 1e-10)))
    
    return -np.sum(radii) + lam * penalty


def _constraints_func(x):
    """Constraint function: all values >= 0 means feasible."""
    x_c = x[0::3]
    y_c = x[1::3]
    r = x[2::3]
    
    cons = []
    # Boundary constraints
    for i in range(N):
        cons.append(x_c[i] - r[i])
        cons.append(1 - x_c[i] - r[i])
        cons.append(y_c[i] - r[i])
        cons.append(1 - y_c[i] - r[i])
    
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            cons.append(dist_sq - min_dist * min_dist)
    
    return np.array(cons)


def _jacobian_constraints(x):
    """Jacobian of constraint function."""
    x_c = x[0::3]
    y_c = x[1::3]
    r = x[2::3]
    
    n_cons = 4 * N + N * (N - 1) // 2
    jac = np.zeros((n_cons, 3 * N))
    
    idx = 0
    for i in range(N):
        # x_i - r_i >= 0
        jac[idx, 3*i] = 1.0
        jac[idx, 3*i+2] = -1.0
        idx += 1
        # 1 - x_i - r_i >= 0
        jac[idx, 3*i] = -1.0
        jac[idx, 3*i+2] = -1.0
        idx += 1
        # y_i - r_i >= 0
        jac[idx, 3*i+1] = 1.0
        jac[idx, 3*i+2] = -1.0
        idx += 1
        # 1 - y_i - r_i >= 0
        jac[idx, 3*i+1] = -1.0
        jac[idx, 3*i+2] = -1.0
        idx += 1
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            min_dist = r[i] + r[j]
            
            jac[idx, 3*i] = 2 * dx
            jac[idx, 3*j] = -2 * dx
            jac[idx, 3*i+1] = 2 * dy
            jac[idx, 3*j+1] = -2 * dy
            jac[idx, 3*i+2] = -2 * min_dist
            jac[idx, 3*j+2] = -2 * min_dist
            idx += 1
    
    return jac


def _obj_simple(x):
    return -np.sum(x[2::3])


def _obj_grad(x):
    g = np.zeros(3 * N)
    g[2::3] = -1.0
    return g


def _make_hex_rows():
    """Hexagonal rows: alternating 5 and 4 circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.096
    y_spacing = r * np.sqrt(3)
    
    total_height = 4 * y_spacing + 2 * r
    y_start = (1.0 - total_height) / 2.0
    
    x_start_5 = (1.0 - 4 * 2 * r) / 2.0
    x_start_4 = (1.0 - 3 * 2 * r) / 2.0
    
    idx = 0
    for row in range(5):
        if row % 2 == 0:
            count = 5
            x_start = x_start_5
        else:
            count = 4
            x_start = x_start_4
        for i in range(count):
            pos[idx, 0] = x_start + i * 2 * r
            pos[idx, 1] = y_start + r + row * y_spacing
            radii[idx] = r
            idx += 1
    
    # 26th circle: find best gap position
    best_score = -1
    best_pos = (0.5, 0.5)
    for cx in np.linspace(0.03, 0.97, 20):
        for cy in np.linspace(0.03, 0.97, 20):
            min_clearance = 1.0
            for i in range(25):
                dist = np.sqrt((cx - pos[i, 0])**2 + (cy - pos[i, 1])**2)
                clearance = dist - radii[i]
                min_clearance = min(min_clearance, clearance)
            min_clearance = min(min_clearance, cx, 1-cx, cy, 1-cy)
            if min_clearance > best_score:
                best_score = min(min_clearance, 0.5)
                best_pos = (cx, cy)
    
    pos[25, 0] = best_pos[0]
    pos[25, 1] = best_pos[1]
    radii[25] = max(best_score * 0.85, 0.015)
    
    return pos, radii


def _make_corner_heavy():
    """Corner-heavy layout: larger circles in corners, smaller in center."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corner_r = 0.14
    corners = [(corner_r, corner_r), (1-corner_r, corner_r), 
               (corner_r, 1-corner_r), (1-corner_r, 1-corner_r)]
    for i in range(4):
        pos[i, 0] = corners[i][0]
        pos[i, 1] = corners[i][1]
        radii[i] = corner_r
    
    edge_r = 0.10
    edges = [(0.3, edge_r), (0.7, edge_r), (0.3, 1-edge_r), (0.7, 1-edge_r),
             (edge_r, 0.3), (1-edge_r, 0.3), (edge_r, 0.7), (1-edge_r, 0.7)]
    for i in range(8):
        idx = 4 + i
        pos[idx, 0] = edges[i][0]
        pos[idx, 1] = edges[i][1]
        radii[idx] = edge_r
    
    diag_r = 0.075
    diags = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.25), (0.5, 0.75)]
    for i in range(6):
        idx = 12 + i
        pos[idx, 0] = diags[i][0]
        pos[idx, 1] = diags[i][1]
        radii[idx] = diag_r
    
    inner_r = 0.055
    inner_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    inner_dist = 0.16
    for i in range(8):
        idx = 18 + i
        pos[idx, 0] = 0.5 + inner_dist * np.cos(inner_angles[i])
        pos[idx, 1] = 0.5 + inner_dist * np.sin(inner_angles[i])
        radii[idx] = inner_r
    
    return pos, radii


def _make_grid():
    """5x5 grid layout with 26th in a corner."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.097
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    pos[25, 0] = 0.08
    pos[25, 1] = 0.92
    radii[25] = 0.07
    return pos, radii


def _make_ring():
    """Central circle with surrounding ring."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0, 0] = 0.5
    pos[0, 1] = 0.5
    radii[0] = 0.11
    
    ring_r = 0.075
    ring_dist = 0.11 + ring_r
    for i in range(12):
        idx = 1 + i
        angle = 2 * np.pi * i / 12
        pos[idx, 0] = 0.5 + ring_dist * np.cos(angle)
        pos[idx, 1] = 0.5 + ring_dist * np.sin(angle)
        radii[idx] = ring_r
    
    corner_r = 0.055
    corners = [(0.07, 0.07), (0.93, 0.07), (0.07, 0.93), (0.93, 0.93),
               (0.5, 0.04), (0.5, 0.96), (0.04, 0.5), (0.96, 0.5),
               (0.22, 0.22), (0.78, 0.22), (0.22, 0.78), (0.78, 0.78),
               (0.5, 0.3)]
    for i in range(13):
        idx = 13 + i
        pos[idx, 0] = corners[i][0]
        pos[idx, 1] = corners[i][1]
        radii[idx] = corner_r
    
    return pos, radii


def _make_random(seed):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.12, 0.88, size=(N, 2))
    radii = rng.uniform(0.05, 0.11, size=N)
    return pos, radii


def _to_xvec(pos, radii):
    """Convert pos (n,2) and radii (n,) to x vector of length 3n."""
    x = np.zeros(3 * N)
    x[0::3] = pos[:, 0]
    x[1::3] = pos[:, 1]
    x[2::3] = radii
    return x


def _from_xvec(x):
    """Convert x vector to pos and radii."""
    pos = np.column_stack([x[0::3], x[1::3]])
    radii = x[2::3].copy()
    return pos, radii


def _trust_constr_optimize(x0, deadline, maxiter=500):
    """Optimize using trust-constr with explicit nonlinear constraints."""
    if time.time() > deadline - 10:
        return None, None, -np.inf
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.001, 0.4))
    
    constraint = NonlinearConstraint(
        _constraints_func,
        0, np.inf,
        jac=_jacobian_constraints
    )
    
    try:
        res = minimize(
            _obj_simple, x0, method='trust-constr',
            jac=_obj_grad, bounds=bounds,
            constraints=[constraint],
            options={'maxiter': maxiter, 'gtol': 1e-9, 'xtol': 1e-12, 'verbose': 0}
        )
        
        x_opt = res.x
        pos_opt, radii_opt = _from_xvec(x_opt)
        radii_opt = np.maximum(radii_opt, 1e-6)
        
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp, score
    except Exception:
        pass
    
    return None, None, -np.inf


def _slsqp_optimize(x0, deadline, maxiter=800):
    """Optimize using SLSQP with explicit constraints."""
    if time.time() > deadline - 10:
        return None, None, -np.inf
    
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N + [(0.001, 0.4)] * N
    
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1] - x[2*N+i]})
    
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: 
                    (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*N+i] + x[2*N+j])**2
            })
    
    def obj(x):
        return -np.sum(x[2*N:])
    
    try:
        res = minimize(
            obj, x0, method='SLSQP', bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-14}
        )
        
        x_opt = res.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x_opt[2*N:], 1e-6)
        
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp, score
    except Exception:
        pass
    
    return None, None, -np.inf


def _local_refine_penalty(pos, radii, deadline):
    """Refine using penalty-based local optimizers."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.3)] * N
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    methods = [
        ('L-BFGS-B', {'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10}),
        ('Powell', {'maxiter': 3000, 'ftol': 1e-14}),
        ('L-BFGS-B', {'maxiter': 4000, 'ftol': 1e-15, 'gtol': 1e-12}),
    ]
    
    for method, opts in methods:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(_objective_penalty, x0, method=method, bounds=bounds, options=opts, args=(500.0,))
            if res.fun < _objective_penalty(x0, 500.0) - 1e-15:
                x_opt = res.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-6)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def _global_search(deadline):
    """Differential evolution global search."""
    bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.25)] * N
    de_deadline = deadline - 25
    
    def wrapper(x):
        return _objective_penalty(x, 300.0)
    
    try:
        res = differential_evolution(
            wrapper, bounds,
            maxiter=150, popsize=20,
            tol=1e-7, mutation=(0.5, 1.5), recombination=0.8,
            seed=7, polish=False, updating='deferred',
            callback=lambda xk, conv: (time.time() > de_deadline)
        )
        x_opt = res.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x_opt[2*N:], 1e-5)
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp, score
    except Exception:
        pass
    
    return None, None, -np.inf


def _high_precision_polish(pos, radii, deadline):
    """Final high-precision polish using trust-constr with tight tolerances."""
    if time.time() > deadline - 5:
        return pos.copy(), radii.copy(), np.sum(radii)
    
    x0 = _to_xvec(pos, radii)
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.001, 0.4))
    
    constraint = NonlinearConstraint(
        _constraints_func,
        0, np.inf,
        jac=_jacobian_constraints
    )
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Multiple trust-constr passes with decreasing tolerances
    for maxiter, gtol, xtol in [(200, 1e-10, 1e-13), (100, 1e-11, 1e-14), (50, 1e-12, 1e-15)]:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                _obj_simple, x0, method='trust-constr',
                jac=_obj_grad, bounds=bounds,
                constraints=[constraint],
                options={'maxiter': maxiter, 'gtol': gtol, 'xtol': xtol, 'verbose': 0}
            )
            
            x_opt = res.x
            pos_opt, radii_opt = _from_xvec(x_opt)
            radii_opt = np.maximum(radii_opt, 1e-6)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-14:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                x0 = _to_xvec(best_pos, best_radii)
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def construct():
    start = time.time()
    deadline = start + 148.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build diverse initial configurations
    configs = []
    p, r = _make_hex_rows()
    configs.append((p, r))
    p, r = _make_grid()
    configs.append((p, r))
    p, r = _make_ring()
    configs.append((p, r))
    p, r = _make_corner_heavy()
    configs.append((p, r))
    for s in [1, 7, 42, 99, 200]:
        p, r = _make_random(s)
        configs.append((p, r))
    
    # Perturbed versions
    rng = np.random.RandomState(123)
    for base_idx in [0, 1, 3]:
        p, r = configs[base_idx]
        pp = p + rng.normal(0, 0.012, size=p.shape)
        pp = np.clip(pp, 0.03, 0.97)
        configs.append((pp, r.copy()))
    
    # Phase 1: Local refinement from each config
    time_per_config = max(8, (deadline - 50) / len(configs))
    
    for idx, (pos, radii) in enumerate(configs):
        t_rem = deadline - time.time()
        if t_rem < 25:
            break
        alloc = min(time_per_config, t_rem - 20)
        if alloc < 6:
            continue
        
        # Penalty-based local refine
        pos_opt, radii_opt, score = _local_refine_penalty(pos, radii, start + time.time() - start + alloc)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Phase 2: Global search if time allows
    if time.time() < deadline - 35 and best_score < 2.3:
        de_pos, de_radii, de_score = _global_search(deadline)
        if de_pos is not None and de_score > best_score + 1e-12:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
    
    # Fallback
    if best_pos is None or best_score < 0:
        p, r = _make_hex_rows()
        p, r = _post_process(p, r)
        best_pos = p
        best_radii = r
        best_score = np.sum(r)
    
    # Phase 3: Trust-constr refinement with explicit constraints
    if time.time() < deadline - 25:
        x0 = _to_xvec(best_pos, best_radii)
        tc_pos, tc_radii, tc_score = _trust_constr_optimize(x0, deadline, maxiter=400)
        if tc_pos is not None and tc_score > best_score + 1e-12:
            best_score = tc_score
            best_pos = tc_pos.copy()
            best_radii = tc_radii.copy()
    
    # Phase 4: SLSQP refinement
    if time.time() < deadline - 20:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        sl_pos, sl_radii, sl_score = _slsqp_optimize(x0, deadline, maxiter=600)
        if sl_pos is not None and sl_score > best_score + 1e-12:
            best_score = sl_score
            best_pos = sl_pos.copy()
            best_radii = sl_radii.copy()
    
    # Phase 5: Another trust-constr pass
    if time.time() < deadline - 15:
        x0 = _to_xvec(best_pos, best_radii)
        tc_pos, tc_radii, tc_score = _trust_constr_optimize(x0, deadline, maxiter=300)
        if tc_pos is not None and tc_score > best_score + 1e-12:
            best_score = tc_score
            best_pos = tc_pos.copy()
            best_radii = tc_radii.copy()
    
    # Phase 6: High-precision final polish
    if time.time() < deadline - 8:
        best_pos, best_radii, best_score = _high_precision_polish(best_pos, best_radii, deadline)
    
    # Final validity check and post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
