# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist[mask] < (min_dist[mask] - tol)):
        return False
    return True

def _post_process(pos, radii, iterations=500):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(iterations):
        valid = True
        
        # Boundary constraints
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999999, 1e-8)
                valid = False
        
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if sum_r > 1e-12:
                        k = dist / sum_r * 0.9999999
                        radii[i] = max(radii[i] * k, 1e-8)
                        radii[j] = max(radii[j] * k, 1e-8)
                        valid = False
                    else:
                        radii[i] = max(radii[i] * 0.99, 1e-8)
                        radii[j] = max(radii[j] * 0.99, 1e-8)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _so_constraints(x, N):
    """Compute SOCP-style constraints: ||p_i - p_j|| >= r_i + r_j and boundary."""
    p = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    # Boundary constraints (must be >= 0)
    c_bound = np.concatenate([
        p[:, 0] - r,
        1.0 - p[:, 0] - r,
        p[:, 1] - r,
        1.0 - p[:, 1] - r
    ])
    
    # SOCP overlap constraints: ||p_i - p_j|| - (r_i + r_j) >= 0
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    i_idx, j_idx = np.where(mask)
    dx = p[i_idx, 0] - p[j_idx, 0]
    dy = p[i_idx, 1] - p[j_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    c_overlap = dist - (r[i_idx] + r[j_idx])
    
    return np.concatenate([c_bound, c_overlap])

def _solve_slsqp(pos, radii, maxiter=300, ftol=1e-14):
    """Solve using SLSQP with SOCP constraint formulation."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = [(1e-4, 0.9999)] * (2 * N) + [(1e-4, 0.25)] * N
    
    def obj(x):
        return -np.sum(x[2*N:])
    
    def cons_fun(x):
        return _so_constraints(x, N)
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': cons_fun}],
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.success or res.fun < -0.5:
            p = res.x[:2*N].reshape(N, 2)
            r = res.x[2*N:]
            r = np.maximum(r, 1e-8)
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                return p, r
    except Exception:
        pass
    return None

def _solve_socp_fixed_positions(pos, maxiter=200):
    """Given fixed positions, maximize sum of radii subject to constraints.
    
    For fixed positions, the problem becomes:
    maximize sum(r_i)
    subject to:
      r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for all i
      r_i + r_j <= ||p_i - p_j|| for all i < j
      r_i >= 0
    
    This is a linear program! We solve it using scipy.optimize.linprog.
    """
    n = N
    # Variables: r_0, r_1, ..., r_{n-1}
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(max_r)
    
    # Overlap constraints: r_i + r_j <= dist(i,j)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(dist)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    bounds = [(1e-8, 0.5)] * n
    
    try:
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            r = res.x
            r = np.maximum(r, 1e-8)
            # Verify validity
            if _is_valid(pos, r):
                return r
            else:
                # Post-process
                pos_copy, r_copy = _post_process(pos.copy(), r.copy())
                if _is_valid(pos_copy, r_copy):
                    return r_copy
    except Exception:
        pass
    return None

def _grow_radii(pos, radii, deadline, max_iter=50):
    """Iteratively grow radii by solving LP for fixed positions, then re-optimize positions."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Step 1: For fixed positions, find optimal radii via LP
        r_opt = _solve_socp_fixed_positions(best_pos)
        if r_opt is not None:
            new_score = np.sum(r_opt)
            if new_score > best_score:
                best_radii = r_opt.copy()
                best_score = new_score
        
        # Step 2: Re-optimize positions with current radii using SLSQP
        res = _solve_slsqp(best_pos, best_radii, maxiter=150, ftol=1e-13)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p.copy()
                best_radii = r.copy()
    
    return best_pos, best_radii

def _init_hexagonal():
    """Hexagonal packing initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.075
    dx = 2 * r
    dy = 2 * r * np.sqrt(3) / 2
    
    idx = 0
    for row in range(6):
        for col in range(6):
            if idx >= N:
                break
            x = 0.5 + col * dx + (row % 2) * dx / 2
            y = 0.5 + row * dy
            x = x - (5 * dx + dx/2) / 2 + 0.5
            y = y - (5 * dy) / 2 + 0.5
            x = np.clip(x, r + 0.005, 1 - r - 0.005)
            y = np.clip(y, r + 0.005, 1 - r - 0.005)
            pos[idx] = [x, y]
            radii[idx] = r
            idx += 1
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_grid():
    """Grid initialization with staggered rows."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.08
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx >= N:
                break
            x = (j + 0.5) / 5.0 + (0.04 if i % 2 == 0 else -0.04)
            y = (i + 0.5) / 5.0
            x = np.clip(x, r + 0.005, 1 - r - 0.005)
            y = np.clip(y, r + 0.005, 1 - r - 0.005)
            pos[idx] = [x, y]
            radii[idx] = r
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.08, 0.92, (N, 2))
    radii = rng.uniform(0.06, 0.09, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_low_discrepancy(seed=42):
    """Initialize using low-discrepancy sequence."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(N):
        t = (i + 0.5) / N
        x = (t * golden_ratio) % 1.0
        y = (t * golden_ratio * golden_ratio) % 1.0
        x = 0.08 + 0.84 * x
        y = 0.08 + 0.84 * y
        pos[i] = [x, y]
        radii[i] = 0.075
    
    pos += rng.normal(0, 0.015, (N, 2))
    pos = np.clip(pos, 0.05, 0.95)
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _local_refine(pos, radii, deadline, rng_seed=456):
    """Iterative local refinement with decreasing perturbation."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(rng_seed)
    
    scales = [0.02, 0.01, 0.005, 0.002, 0.001]
    
    for scale in scales:
        while time.time() < deadline - 8:
            perturbed_pos = best_pos + rng.normal(0, scale, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.01, 0.99)
            perturbed_radii = best_radii * (1 + rng.normal(0, 0.02, N))
            perturbed_radii = np.clip(perturbed_radii, 0.015, 0.22)
            
            perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
            
            res = _solve_slsqp(perturbed_pos, perturbed_radii, maxiter=150, ftol=1e-13)
            if res:
                p, r = res
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_pos = p.copy()
                    best_radii = r.copy()
    
    return best_pos, best_radii

def _global_search(pos_init, radii_init, deadline):
    """Use differential evolution for global search."""
    base_r = np.mean(radii_init)
    
    def de_obj(x):
        p = x[:2*N].reshape(N, 2)
        p = np.clip(p, 0.02, 0.98)
        r = np.full(N, base_r)
        p, r = _post_process(p, r)
        if not _is_valid(p, r):
            return 1000
        return -np.sum(r)
    
    bounds = [(0.02, 0.98)] * (2 * N)
    
    try:
        result = differential_evolution(
            de_obj, bounds, 
            seed=789, maxiter=30, popsize=10,
            tol=1e-5, mutation=(0.4, 1.2), recombination=0.7,
            polish=True, updating='deferred'
        )
        if result.success:
            p = result.x[:2*N].reshape(N, 2)
            r = np.full(N, base_r)
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                res = _solve_slsqp(p, r, maxiter=200, ftol=1e-13)
                if res:
                    return res
                return p, r
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_pos = None
    best_radii = None
    best_score = -1

    # 1. Try multiple initializations with SLSQP
    inits = []
    inits.append(_init_hexagonal())
    inits.append(_init_grid())
    inits.append(_init_low_discrepancy(42))
    inits.append(_init_low_discrepancy(123))
    inits.append(_init_random(1))
    inits.append(_init_random(2))
    inits.append(_init_random(3))
    inits.append(_init_random(4))
    
    for pos, radii in inits:
        if time.time() > deadline - 30:
            break
        res = _solve_slsqp(pos, radii, maxiter=200, ftol=1e-13)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 2. Global search with differential evolution
    if time.time() < deadline - 40:
        init_pos, init_radii = _init_grid()
        res = _global_search(init_pos, init_radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 3. Grow radii using alternating LP + SLSQP
    if best_pos is not None and time.time() < deadline - 15:
        grown_pos, grown_radii = _grow_radii(best_pos, best_radii, deadline, max_iter=30)
        s = np.sum(grown_radii)
        if s > best_score:
            best_score = s
            best_pos = grown_pos
            best_radii = grown_radii

    # 4. Local refinement on best solution
    if best_pos is not None and time.time() < deadline - 10:
        refined_pos, refined_radii = _local_refine(best_pos, best_radii, deadline, rng_seed=456)
        s = np.sum(refined_radii)
        if s > best_score:
            best_score = s
            best_pos = refined_pos
            best_radii = refined_radii

    # 5. Final grow + refine
    if best_pos is not None and time.time() < deadline - 5:
        grown_pos, grown_radii = _grow_radii(best_pos, best_radii, deadline, max_iter=20)
        s = np.sum(grown_radii)
        if s > best_score:
            best_score = s
            best_pos = grown_pos
            best_radii = grown_radii

    # 6. Additional refinement passes
    if best_pos is not None and time.time() < deadline - 3:
        for seed in [789, 1024]:
            if time.time() > deadline - 2:
                break
            refined_pos, refined_radii = _local_refine(best_pos, best_radii, deadline, rng_seed=seed)
            s = np.sum(refined_radii)
            if s > best_score:
                best_score = s
                best_pos = refined_pos
                best_radii = refined_radii

    if best_pos is None:
        pos, radii = _init_grid()
        pos, radii = _post_process(pos, radii)
        return np.column_stack([pos, radii])

    # Final validity check
    if not _is_valid(best_pos, best_radii):
        best_pos, best_radii = _post_process(best_pos, best_radii)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
