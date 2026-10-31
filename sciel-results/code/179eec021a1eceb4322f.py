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
    """Shrink radii slightly to ensure strict validity using vectorized checks."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(20):
        # Check boundary violations
        min_dist_to_wall = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]), 
                                      np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        wall_violation = radii > min_dist_to_wall + tol
        if np.any(wall_violation):
            # Scale down violating radii
            factor = np.where(wall_violation, (min_dist_to_wall - tol) / (radii + 1e-12), 1.0)
            factor = np.maximum(factor, 0.99)
            radii = radii * factor
            radii = np.maximum(radii, 1e-5)
            continue
        
        # Check overlap violations
        # dist_sq[i,j] - (r_i + r_j)^2 >= 0
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Only check upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        violations = (dist_sq - sum_r * sum_r) < -tol
        
        if np.any(violations):
            # Find the worst violation to determine a global shrink factor
            # Or shrink all pairs involved? 
            # A simple global shrink is safer but might be too aggressive.
            # Let's find the maximum violation magnitude.
            viol_mag = np.where(violations, sum_r * sum_r - dist_sq, 0)
            max_viol = np.max(viol_mag)
            if max_viol > 0:
                # Shrink all radii by a small factor proportional to violation
                # This is a heuristic. A better way is to shrink only the involved circles.
                # For speed, we'll do a global shrink if any violation exists.
                shrink_factor = 1.0 - 0.01 * np.log1p(max_viol)
                shrink_factor = max(shrink_factor, 0.95)
                radii = radii * shrink_factor
                radii = np.maximum(radii, 1e-5)
            continue
            
        # If no violations, we are valid
        break
        
    # Final strict check and tiny shrink for safety
    radii = radii * 0.99999
    radii = np.maximum(radii, 1e-5)
    
    return pos, radii

def _initial_greedy_best_fit():
    """Generate a tight initial packing using a greedy best-fit algorithm with grid refinement."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # First circle: center
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        # Use a coarser grid for the first few circles to establish structure
        # Then switch to random sampling
        if i < 5:
            # Grid search
            step = 0.05
            for x in np.arange(0.05, 0.95, step):
                for y in np.arange(0.05, 0.95, step):
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
        else:
            # Random sampling
            rng = np.random.RandomState(i * 1000 + 42)
            for _ in range(3000):
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
    Vectorized for speed.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Boundary constraints (4N values)
    b_cons = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    
    # Overlap constraints - vectorized
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
    o_cons_full = dist_sq - sum_r * sum_r
    # Take upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    o_cons = o_cons_full[mask]
    
    all_cons = np.concatenate([b_cons, o_cons])
    
    # Soft penalty: soft_min(0, c) = -log(1 + exp(-beta*c))/beta
    scaled = -beta * all_cons
    scaled = np.clip(scaled, -500, 500)
    penalty = -np.log1p(np.exp(scaled)) / beta
    
    return -np.sum(radii) + penalty.sum()

def _hard_constraints(x):
    """All hard inequality constraints (for SLSQP). Vectorized."""
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    b_cons = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
    o_cons_full = dist_sq - sum_r * sum_r
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    o_cons = o_cons_full[mask]
    
    return np.concatenate([b_cons, o_cons])

def _high_precision_polish(pos, radii, max_iters=200, time_limit=None, start_time=None):
    """
    High-precision polish: alternate between SLSQP with tight tolerance,
    force-directed relaxation, and gradient-based radius expansion.
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
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            if result.success or result.fun < -np.sum(radii) - 1e-10:
                pos_new = result.x[:2*N].reshape(N, 2)
                radii_new = result.x[2*N:]
                pos_new, radii_new = _post_process(pos_new, radii_new, tol=1e-9)
                if _is_valid(pos_new, radii_new, tol=1e-8):
                    score = np.sum(radii_new)
                    if score > best_score + 1e-12:
                        best_score = score
                        best_pos = pos_new.copy()
                        best_radii = radii_new.copy()
                        pos = pos_new
                        radii = radii_new
                    else:
                        break
        except Exception:
            break
        
        # Force-directed micro-relaxation
        if time.time() - start_time > time_limit:
            break
        try:
            pos_fd, radii_fd = _force_directed_relax(
                pos, radii,
                n_iters=50, dt=0.002,
                time_limit=min(3.0, time_limit - (time.time() - start_time)),
                start_time=start_time
            )
            if _is_valid(pos_fd, radii_fd, tol=1e-8):
                score = np.sum(radii_fd)
                if score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
                    pos = pos_fd
                    radii = radii_fd
        except Exception:
            pass
        
        # Gradient-based nudge: try increasing radii slightly and re-solve
        if time.time() - start_time > time_limit:
            break
        try:
            radii_try = radii * 1.002 + 1e-6
            x_try = np.concatenate([pos.flatten(), radii_try])
            result2 = minimize(
                lambda x: -np.sum(x[2*N:]),
                x_try, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': _hard_constraints},
                options={'maxiter': 50, 'ftol': 1e-14}
            )
            pos_try = result2.x[:2*N].reshape(N, 2)
            radii_try_out = result2.x[2*N:]
            pos_try, radii_try_out = _post_process(pos_try, radii_try_out, tol=1e-9)
            if _is_valid(pos_try, radii_try_out, tol=1e-8):
                score = np.sum(radii_try_out)
                if score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_try.copy()
                    best_radii = radii_try_out.copy()
                    pos = pos_try
                    radii = radii_try_out
        except Exception:
            pass
    
    return best_pos, best_radii

def _force_directed_relax(pos, radii, n_iters=200, dt=0.01, time_limit=None, start_time=None):
    """
    Force-directed relaxation: treat overlapping circles as repelling,
    and gently push toward expanding all radii. Exploits contact structure.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 15.0
    
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros((N, 2))
        r_forces = np.zeros(N)
        
        # Pairwise overlap repulsion
        dx = pos[i_idx, 0] - pos[j_idx, 0]
        dy = pos[i_idx, 1] - pos[j_idx, 1]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        sum_r = radii[i_idx] + radii[j_idx]
        overlap = sum_r - dist  # positive if overlapping
        
        # Only apply force where overlapping
        mask = overlap > 1e-8
        if np.any(mask):
            f_mag = overlap[mask]
            # Direction: push apart
            dir_x = dx[mask] / dist[mask]
            dir_y = dy[mask] / dist[mask]
            
            # Scatter forces
            np.add.at(forces, i_idx[mask], -f_mag[:, np.newaxis] * np.stack([dir_x, dir_y], axis=1))
            np.add.at(forces, j_idx[mask], f_mag[:, np.newaxis] * np.stack([dir_x, dir_y], axis=1))
            
            # Also try to shrink the larger circle slightly
            r_i = radii[i_idx[mask]]
            r_j = radii[j_idx[mask]]
            r_forces[i_idx[mask]] -= f_mag * 0.5
            r_forces[j_idx[mask]] -= f_mag * 0.5
        
        # Boundary forces: push away from walls
        wall_f = np.zeros((N, 2))
        for i in range(N):
            r = radii[i]
            if pos[i, 0] < r:
                wall_f[i, 0] = (r - pos[i, 0])
            elif pos[i, 0] > 1.0 - r:
                wall_f[i, 0] = -(pos[i, 0] - (1.0 - r))
            if pos[i, 1] < r:
                wall_f[i, 1] = (r - pos[i, 1])
            elif pos[i, 1] > 1.0 - r:
                wall_f[i, 1] = -(pos[i, 1] - (1.0 - r))
        
        # Gentle expansion pressure
        r_forces += 0.001
        
        # Update
        learning_rate = dt * (0.99 ** it)
        pos += learning_rate * (forces + wall_f)
        radii += learning_rate * r_forces
        
        # Clamp
        pos[:, 0] = np.clip(pos[:, 0], 0, 1)
        pos[:, 1] = np.clip(pos[:, 1], 0, 1)
        radii = np.maximum(radii, 1e-5)
        
        # Post-process to ensure validity
        pos, radii = _post_process(pos, radii, tol=1e-8)
        
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
    
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
    then force-directed relaxation, then SLSQP polish.
    """
    x0 = np.concatenate([pos_init.flatten(), radii_init])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    best_pos = pos_init.copy()
    best_radii = radii_init.copy()
    best_score = np.sum(radii_init)
    
    # Phase 1: L-BFGS-B with soft penalty, increasing beta
    betas = [20.0, 50.0, 100.0, 200.0]
    for beta in betas:
        if time.time() > deadline - 15:
            break
        try:
            def obj(x, b=beta):
                return _soft_penalty_objective(x, beta=b)
            
            result = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pos_l = result.x[:2*N].reshape(N, 2)
            radii_l = result.x[2*N:]
            pos_l, radii_l = _post_process(pos_l, radii_l)
            if _is_valid(pos_l, radii_l):
                score = np.sum(radii_l)
                if score > best_score:
                    best_score = score
                    best_pos = pos_l.copy()
                    best_radii = radii_l.copy()
                    x0 = result.x
        except Exception:
            pass
    
    # Phase 2: Force-directed relaxation to exploit contact structure
    if time.time() < deadline - 10:
        try:
            pos_fd, radii_fd = _force_directed_relax(
                best_pos, best_radii,
                n_iters=150, dt=0.005,
                time_limit=min(10.0, deadline - time.time() - 8),
                start_time=start_time
            )
            if _is_valid(pos_fd, radii_fd):
                score = np.sum(radii_fd)
                if score > best_score:
                    best_score = score
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
        except Exception:
            pass
    
    # Phase 3: SLSQP hard constraints for polish
    if time.time() < deadline - 5:
        try:
            result2 = minimize(
                lambda x: -np.sum(x[2*N:]),
                np.concatenate([best_pos.flatten(), best_radii]),
                method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': _hard_constraints},
                options={'maxiter': 80, 'ftol': 1e-12}
            )
            pos_s = result2.x[:2*N].reshape(N, 2)
            radii_s = result2.x[2*N:]
            pos_s, radii_s = _post_process(pos_s, radii_s)
            if _is_valid(pos_s, radii_s):
                score = np.sum(radii_s)
                if score > best_score:
                    best_score = score
                    best_pos = pos_s.copy()
                    best_radii = radii_s.copy()
        except Exception:
            pass
    
    return best_pos, best_radii

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
        if time.time() > deadline - 15:
            break
        
        try:
            pos_init, rad_init = init_func()
        except Exception:
            continue
        
        # Strategy B: Soft penalty L-BFGS-B then force-directed then SLSQP polish
        if time.time() < deadline - 10:
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
        if time.time() < deadline - 5 and best_pos is not None:
            try:
                pos_ref, rad_ref = _vectorized_refine(
                    best_pos, best_radii,
                    steps=80, step_size=0.008,
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
        
        # Force-directed on best so far (additional pass)
        if time.time() < deadline - 5 and best_pos is not None:
            try:
                pos_fd, rad_fd = _force_directed_relax(
                    best_pos, best_radii,
                    n_iters=100, dt=0.003,
                    time_limit=min(5.0, deadline - time.time() - 4),
                    start_time=start_time
                )
                if _is_valid(pos_fd, rad_fd):
                    score_fd = np.sum(rad_fd)
                    if score_fd > best_score:
                        best_score = score_fd
                        best_pos = pos_fd.copy()
                        best_radii = rad_fd.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if best_pos is not None and time.time() < deadline - 3:
        try:
            pos_pol, rad_pol = _high_precision_polish(
                best_pos, best_radii,
                max_iters=50,
                time_limit=deadline - time.time() - 2,
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
