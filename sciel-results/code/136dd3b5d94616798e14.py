"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _objective(x):
    """
    Objective function: minimize negative sum of radii.
    x is a flattened array of [x_1, ..., x_N, y_1, ..., y_N, r_1, ..., r_N]
    """
    radii = x[2*N:]
    return -np.sum(radii)

def _boundary_constraints(x):
    """
    Boundary constraints:
    x_i - r_i >= 0
    1 - x_i - r_i >= 0
    y_i - r_i >= 0
    1 - y_i - r_i >= 0
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    cons = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    return cons

def _overlap_constraints(x):
    """
    Overlap constraints:
    (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    cons = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            cons.append(dist_sq - sum_r * sum_r)
            
    return np.array(cons)

def _all_constraints(x):
    """Combine all constraints into a single array."""
    b_cons = _boundary_constraints(x)
    o_cons = _overlap_constraints(x)
    return np.concatenate([b_cons, o_cons])

def _soft_penalty_objective(x, beta=50.0):
    """
    Smooth penalty-based objective: minimize -sum(radii) + penalty for constraint violations.
    Uses log-sum-exp style soft-min for smoothness.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Boundary violations (should be >= 0)
    b_cons = _boundary_constraints(x)
    # Overlap violations (should be >= 0)
    o_cons = _overlap_constraints(x)
    
    all_cons = np.concatenate([b_cons, o_cons])
    
    # Soft penalty: sum of soft-min(0, c) for each constraint c
    # soft_min(0, c) = -softmax(-c, 0) / beta
    # For a single value: soft_min(0, c) ≈ -log(1 + exp(-beta*c)) / beta for c near 0
    # More numerically stable: 
    # if c >= 0: penalty ≈ 0
    # if c < 0: penalty ≈ -c (for large beta)
    
    # Use: penalty = sum(softplus(-beta * c)) / beta where softplus(z) = log(1+exp(z))
    # This is smooth and approximates max(0, -c) for large beta
    
    z = -beta * all_cons
    # softplus(z) = log(1 + exp(z)), computed stably
    # For z > 0: softplus(z) ≈ z + log(1 + exp(-z))
    # For z <= 0: softplus(z) ≈ log(1 + exp(z))
    softplus_vals = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    
    penalty = np.sum(softplus_vals) / beta
    
    return -np.sum(radii) + penalty

def _is_valid(pos, radii, tol=1e-6):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist
    # We use a slightly tighter tolerance for safety
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii, tol=1e-7):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(50):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + tol:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r + tol:
                    # Scale both radii to make them just touch with a small gap
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

def _high_precision_polish(pos, radii, time_budget=5.0):
    """
    High-precision polish pass using SLSQP with tighter tolerances on the
    active constraint set. This is the final refinement step.
    """
    start_t = time.time()
    pos = pos.copy()
    radii = radii.copy()
    
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, pos[i,0] - 0.1), min(1.0, pos[i,0] + 0.1)))
        bounds.append((max(0.0, pos[i,1] - 0.1), min(1.0, pos[i,1] + 0.1)))
    for i in range(N):
        bounds.append((max(1e-5, radii[i] * 0.9), min(0.5, radii[i] * 1.1)))
    
    def constraints_func(x):
        return _all_constraints(x)
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    # First pass: moderate precision
    try:
        result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints_func},
                          options={'maxiter': 100, 'ftol': 1e-10})
        if result.success or result.fun < _objective(x0):
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-8)
            if _is_valid(pos_pp, radii_pp, tol=1e-7):
                pos, radii = pos_pp, radii_pp
    except Exception:
        pass
    
    # Second pass: even tighter, only if time remains
    if time.time() - start_t < time_budget * 0.5:
        try:
            x0 = np.concatenate([pos.flatten(), radii])
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints_func},
                              options={'maxiter': 50, 'ftol': 1e-12})
            if result.success or result.fun < _objective(x0):
                pos_new = result.x[:2*N].reshape(N, 2)
                radii_new = result.x[2*N:]
                pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-9)
                if _is_valid(pos_pp, radii_pp, tol=1e-8):
                    pos, radii = pos_pp, radii_pp
        except Exception:
            pass
    
    return pos, radii

def _initial_greedy_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample 8000 random points for better coverage
        for _ in range(8000):
            x = rng.uniform(0.01, 0.99)
            y = rng.uniform(0.01, 0.99)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            
            # Max radius allowed by existing circles
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
    """
    Generate a hexagonal lattice-like initial state.
    """
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
            
            # Clip to unit square
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            pos[k] = [x, y]
            radii[k] = r
            k += 1
            
    # If we have fewer than N, fill remaining with small circles in corners
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

def _vectorized_refine(pos, radii, steps=500, step_size=0.05, rng=None):
    """
    Vectorized local search to refine the packing.
    Tries to increase radii and move positions to satisfy constraints.
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for step in range(steps):
        # Decay step size
        current_step = step_size * (0.995 ** step)
        
        # Generate random perturbations
        delta_pos = rng.normal(0, current_step, (N, 2))
        delta_radii = rng.normal(0, current_step * 0.3, N)
        
        new_pos = pos + delta_pos
        new_radii = radii + delta_radii
        
        # Clip positions to stay within bounds roughly
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, 1)
        new_radii = np.maximum(new_radii, 1e-5)
        
        # Post-process to ensure validity
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

def _smooth_optimize(pos, radii, beta_start=20.0, beta_end=100.0, n_iters=3):
    """
    Optimize using the smooth penalty objective with increasing beta.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    for it in range(n_iters):
        beta = beta_start * (beta_end / beta_start) ** (it / max(n_iters - 1, 1))
        
        def obj(x):
            return _soft_penalty_objective(x, beta=beta)
        
        x0 = np.concatenate([pos.flatten(), radii])
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 100, 'ftol': 1e-12})
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-8)
            if _is_valid(pos_pp, radii_pp, tol=1e-7):
                pos, radii = pos_pp, radii_pp
        except Exception:
            pass
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # List of initializers to try
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
    
    # Bounds for optimizer
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
        
    # Constraint function for SLSQP
    def constraints_func(x):
        return _all_constraints(x)

    # Try each initializer
    for idx, init_func in enumerate(initializers):
        if time.time() > deadline - 20:
            break
            
        try:
            pos_init, rad_init = init_func()
        except Exception:
            continue
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # First: try smooth penalty optimization (faster, more robust)
        try:
            pos_smooth, radii_smooth = _smooth_optimize(pos_init, rad_init, beta_start=20.0, beta_end=80.0, n_iters=3)
            score_smooth = np.sum(radii_smooth)
            if score_smooth > best_score:
                best_score = score_smooth
                best_pos = pos_smooth.copy()
                best_radii = radii_smooth.copy()
        except Exception:
            pass
        
        # Second: try SLSQP with hard constraints
        try:
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints_func},
                              options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.success or result.fun < _objective(x0):
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = result.x[2*N:]
                
                # Post-process
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
                        
        except Exception:
            pass
            
        # Vectorized refinement on the best result so far
        if time.time() < deadline - 15 and best_pos is not None:
            pos_ref, rad_ref = _vectorized_refine(best_pos, best_radii, steps=400, step_size=0.015, rng=np.random.RandomState(idx * 7 + 1))
            if _is_valid(pos_ref, rad_ref):
                score_ref = np.sum(rad_ref)
                if score_ref > best_score:
                    best_score = score_ref
                    best_pos = pos_ref.copy()
                    best_radii = rad_ref.copy()

    # Final high-precision polish on the best result
    if best_pos is not None and time.time() < deadline - 3:
        pos_polished, radii_polished = _high_precision_polish(best_pos, best_radii, time_budget=4.0)
        if _is_valid(pos_polished, radii_polished, tol=1e-8):
            score_polished = np.sum(radii_polished)
            if score_polished > best_score:
                best_score = score_polished
                best_pos = pos_polished.copy()
                best_radii = radii_polished.copy()

    # Final fallback if no valid result found
    if best_pos is None or not _is_valid(best_pos, best_radii):
        # Simple grid fallback
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
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
