"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-6):
    """Strict validity check for final result."""
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

def _post_process(pos, radii, tol=1e-7):
    """Shrink radii slightly to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(80):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + tol:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r + tol:
                    if sum_r > 1e-9:
                        k = (dist - tol) / (sum_r + 1e-12)
                    else:
                        k = 0.99
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _initial_greedy_best_fit():
    """Generate a tight initial packing using a greedy best-fit algorithm."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        for _ in range(8000):
            x = rng.uniform(0.01, 0.99)
            y = rng.uniform(0.01, 0.99)
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
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
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_hexagonal():
    """Generate a hexagonal lattice-like initial state."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.12
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(5):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            pos[k] = [x, y]
            radii[k] = r
            k += 1
    while k < N:
        if k == 25:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.05
        else:
            pos[k] = [0.1, 0.1]
            radii[k] = 0.05
        k += 1
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _soft_penalty_objective(x, beta=80.0):
    """
    Smooth penalty-based objective using soft-min for constraint violations.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Boundary constraints
    b_cons = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    
    # Overlap constraints
    o_cons_list = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            o_cons_list.append(dist_sq - sum_r * sum_r)
    o_cons = np.array(o_cons_list)
    
    all_cons = np.concatenate([b_cons, o_cons])
    
    # Soft penalty: for each constraint c (should be >= 0),
    # penalty = -log(1 + exp(-beta*c)) / beta
    # For c >> 0: penalty ~ 0
    # For c << 0: penalty ~ -c
    # Numerically stable version:
    # soft_min(0, c) = -log(1 + exp(-beta*c))/beta
    # Use: if c > 20/beta: penalty ~ 0; else compute
    scaled = -beta * all_cons
    # clip to avoid overflow
    scaled = np.clip(scaled, -500, 500)
    penalty = -np.log(1.0 + np.exp(scaled)) / beta
    
    return -np.sum(radii) + penalty.sum()

def _hard_constraints(x):
    """All hard inequality constraints (for SLSQP)."""
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    b_cons = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    
    o_cons_list = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            o_cons_list.append(dist_sq - sum_r * sum_r)
    o_cons = np.array(o_cons_list)
    
    return np.concatenate([b_cons, o_cons])

def _high_precision_polish(pos, radii, max_iters=200, time_limit=None, start_time=None):
    """
    High-precision polish: alternate between SLSQP with tight tolerance
    and a coordinate-descent / gradient step to squeeze out more sum of radii.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 30.0
    
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        x0 = np.concatenate([pos.flatten(), radii])
        
        # SLSQP with very tight tolerance
        try:
            result = minimize(
                lambda x: -np.sum(x[2*N:]),
                x0, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': _hard_constraints},
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if result.success or result.fun < -np.sum(radii):
                pos_new = result.x[:2*N].reshape(N, 2)
                radii_new = result.x[2*N:]
                pos_new, radii_new = _post_process(pos_new, radii_new, tol=1e-8)
                if _is_valid(pos_new, radii_new, tol=1e-7):
                    score = np.sum(radii_new)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_new.copy()
                        best_radii = radii_new.copy()
                        pos = pos_new
                        radii = radii_new
                    else:
                        break
        except Exception:
            break
        
        # Gradient-based nudge: try increasing radii slightly and re-solve
        if time.time() - start_time > time_limit:
            break
        try:
            radii_try = radii * 1.001 + 1e-6
            x_try = np.concatenate([pos.flatten(), radii_try])
            result2 = minimize(
                lambda x: -np.sum(x[2*N:]),
                x_try, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': _hard_constraints},
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            pos_try = result2.x[:2*N].reshape(N, 2)
            radii_try_out = result2.x[2*N:]
            pos_try, radii_try_out = _post_process(pos_try, radii_try_out, tol=1e-8)
            if _is_valid(pos_try, radii_try_out, tol=1e-7):
                score = np.sum(radii_try_out)
                if score > best_score:
                    best_score = score
                    best_pos = pos_try.copy()
                    best_radii = radii_try_out.copy()
                    pos = pos_try
                    radii = radii_try_out
        except Exception:
            pass
    
    return best_pos, best_radii

def _vectorized_refine(pos, radii, steps=400, step_size=0.03, rng=None):
    """
    Vectorized local search to refine the packing.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for step in range(steps):
        current_step = step_size * (0.995 ** step)
        delta_pos = rng.normal(0, current_step, (N, 2))
        delta_radii = rng.normal(0, current_step * 0.3, N)
        new_pos = pos + delta_pos
        new_radii = radii + delta_radii
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, 1)
        new_radii = np.maximum(new_radii, 1e-5)
        new_pos, new_radii = _post_process(new_pos, new_radii, tol=1e-8)
        if _is_valid(new_pos, new_radii, tol=1e-7):
            score = np.sum(new_radii)
            if score > best_score:
                best_score = score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
                pos = new_pos
                radii = new_radii
    return best_pos, best_radii

def _soft_penalty_optimize(pos_init, radii_init, start_time, deadline):
    """
    Optimize using the soft penalty objective with L-BFGS-B,
    then polish with SLSQP.
    """
    x0 = np.concatenate([pos_init.flatten(), radii_init])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    # Phase 1: L-BFGS-B with soft penalty
    try:
        result = minimize(
            _soft_penalty_objective, x0, method='L-BFGS-B',
            bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        pos_l = result.x[:2*N].reshape(N, 2)
        radii_l = result.x[2*N:]
        pos_l, radii_l = _post_process(pos_l, radii_l)
    except Exception:
        pos_l, radii_l = pos_init, radii_init
    
    # Phase 2: SLSQP hard constraints for polish
    if time.time() < deadline - 5:
        try:
            result2 = minimize(
                lambda x: -np.sum(x[2*N:]),
                np.concatenate([pos_l.flatten(), radii_l]),
                method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': _hard_constraints},
                options={'maxiter': 150, 'ftol': 1e-10}
            )
            pos_s = result2.x[:2*N].reshape(N, 2)
            radii_s = result2.x[2*N:]
            pos_s, radii_s = _post_process(pos_s, radii_s)
            if _is_valid(pos_s, radii_s):
                return pos_s, radii_s
        except Exception:
            pass
    
    if _is_valid(pos_l, radii_l):
        return pos_l, radii_l
    return pos_init, radii_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    initializers = [
        _initial_greedy_best_fit,
        _initial_hexagonal,
        lambda: _initial_random(1),
        lambda: _initial_random(2),
        lambda: _initial_random(3),
        lambda: _initial_random(4),
        lambda: _initial_random(5),
        lambda: _initial_random(6),
        lambda: _initial_random(7),
    ]
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    for idx, init_func in enumerate(initializers):
        if time.time() > deadline - 20:
            break
        
        try:
            pos_init, rad_init = init_func()
        except Exception:
            continue
        
        # Strategy A: SLSQP directly
        if time.time() < deadline - 15:
            try:
                x0 = np.concatenate([pos_init.flatten(), rad_init])
                result = minimize(
                    lambda x: -np.sum(x[2*N:]), x0,
                    method='SLSQP', bounds=bounds,
                    constraints={'type': 'ineq', 'fun': _hard_constraints},
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = result.x[2*N:]
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Strategy B: Soft penalty L-BFGS-B then SLSQP polish
        if time.time() < deadline - 15:
            try:
                pos_b, radii_b = _soft_penalty_optimize(pos_init, rad_init, start_time, deadline)
                if _is_valid(pos_b, radii_b):
                    score = np.sum(radii_b)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_b.copy()
                        best_radii = radii_b.copy()
            except Exception:
                pass
        
        # Vectorized refinement on best so far
        if time.time() < deadline - 10 and best_pos is not None:
            try:
                pos_ref, rad_ref = _vectorized_refine(
                    best_pos, best_radii,
                    steps=200, step_size=0.02,
                    rng=np.random.RandomState(idx * 7 + 1)
                )
                if _is_valid(pos_ref, rad_ref):
                    score_ref = np.sum(rad_ref)
                    if score_ref > best_score:
                        best_score = score_ref
                        best_pos = pos_ref.copy()
                        best_radii = rad_ref.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if best_pos is not None and time.time() < deadline - 5:
        try:
            pos_pol, rad_pol = _high_precision_polish(
                best_pos, best_radii,
                max_iters=100,
                time_limit=deadline - time.time() - 3,
                start_time=start_time
            )
            if _is_valid(pos_pol, rad_pol):
                score_pol = np.sum(rad_pol)
                if score_pol > best_score:
                    best_score = score_pol
                    best_pos = pos_pol.copy()
                    best_radii = rad_pol.copy()
        except Exception:
            pass
    
    # Final fallback
    if best_pos is None or not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = _post_process(pos, radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
