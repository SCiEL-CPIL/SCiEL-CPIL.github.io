# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

def _compute_score(radii):
    """Compute the sum of radii, the objective to maximize."""
    return np.sum(radii)

def _is_valid(pos, radii, tol=1e-7):
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

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(300):
        valid = True
        
        # Boundary constraints
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.999999, 1e-6)
                valid = False
        
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if sum_r > 1e-10:
                        k = dist / sum_r * 0.999999
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
                    else:
                        radii[i] = max(radii[i] * 0.99, 1e-6)
                        radii[j] = max(radii[j] * 0.99, 1e-6)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _so_constraints(x, N):
    """Compute SOCP-style constraints: ||p_i - p_j|| >= r_i + r_j and boundary."""
    p = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    # Boundary constraints
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

def _solve_slsqp_so(pos, radii, maxiter=200):
    """Solve using SLSQP with SOCP constraint formulation."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = [(0.01, 0.99)] * (2 * N) + [(0.01, 0.25)] * N
    
    def obj(x):
        return -np.sum(x[2*N:])
    
    def cons_fun(x):
        return _so_constraints(x, N)
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': cons_fun}],
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.success:
            p = res.x[:2*N].reshape(N, 2)
            r = res.x[2*N:]
            r = np.maximum(r, 1e-6)
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                return p, r
    except Exception:
        pass
    return None

def _init_hexagonal():
    """Hexagonal packing initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal lattice
    r = 0.08
    dx = 2 * r
    dy = 2 * r * np.sqrt(3) / 2
    
    idx = 0
    for row in range(6):
        for col in range(6):
            if idx >= N:
                break
            x = 0.5 + col * dx + (row % 2) * dx / 2
            y = 0.5 + row * dy
            # Center in unit square
            x = x - (5 * dx + dx/2) / 2 + 0.5
            y = y - (5 * dy) / 2 + 0.5
            x = np.clip(x, r + 0.01, 1 - r - 0.01)
            y = np.clip(y, r + 0.01, 1 - r - 0.01)
            pos[idx] = [x, y]
            radii[idx] = r
            idx += 1
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_grid():
    """Grid initialization with staggered rows."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid with stagger
    r = 0.085
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx >= N:
                break
            x = (j + 0.5) / 5.0 + (0.05 if i % 2 == 0 else -0.05)
            y = (i + 0.5) / 5.0
            x = np.clip(x, r + 0.01, 1 - r - 0.01)
            y = np.clip(y, r + 0.01, 1 - r - 0.01)
            pos[idx] = [x, y]
            radii[idx] = r
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.07, 0.10, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_from_known(seed=42):
    """Initialize based on known good patterns with perturbation."""
    rng = np.random.RandomState(seed)
    
    # Start with a reasonable packing
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Place circles in a quasi-random pattern
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(N):
        # Use low-discrepancy sequence for initial positions
        t = i / N
        x = (t * golden_ratio) % 1.0
        y = (t * golden_ratio * golden_ratio) % 1.0
        x = 0.1 + 0.8 * x
        y = 0.1 + 0.8 * y
        pos[i] = [x, y]
        radii[i] = 0.08
    
    # Add small perturbation
    pos += rng.normal(0, 0.02, (N, 2))
    pos = np.clip(pos, 0.05, 0.95)
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _local_refine_so(pos, radii, deadline):
    """Iterative local refinement using SOCP formulation."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(456)
    
    # Multiple refinement passes with decreasing perturbation
    for scale in [0.03, 0.02, 0.01, 0.005, 0.002]:
        while time.time() < deadline - 5:
            perturbed_pos = best_pos + rng.normal(0, scale, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.02, 0.98)
            perturbed_radii = best_radii * (1 + rng.normal(0, 0.03, N))
            perturbed_radii = np.clip(perturbed_radii, 0.02, 0.22)
            
            perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
            
            res = _solve_slsqp_so(perturbed_pos, perturbed_radii, maxiter=150)
            if res:
                p, r = res
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_pos = p.copy()
                    best_radii = r.copy()
    
    return best_pos, best_radii

def _global_search_de(pos_init, radii_init, deadline):
    """Use differential evolution for global search on positions."""
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
            seed=789, maxiter=50, popsize=15,
            tol=1e-6, mutation=(0.5, 1.5), recombination=0.8,
            polish=True, updating='deferred'
        )
        if result.success:
            p = result.x[:2*N].reshape(N, 2)
            r = np.full(N, base_r)
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                res = _solve_slsqp_so(p, r, maxiter=200)
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

    # 1. Try multiple initializations with SLSQP SOCP
    inits = []
    inits.append(_init_hexagonal())
    inits.append(_init_grid())
    inits.append(_init_from_known(42))
    inits.append(_init_from_known(123))
    inits.append(_init_random(1))
    inits.append(_init_random(2))
    inits.append(_init_random(3))
    
    for pos, radii in inits:
        if time.time() > deadline - 20:
            break
        res = _solve_slsqp_so(pos, radii, maxiter=200)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 2. Global search with differential evolution
    if time.time() < deadline - 30:
        init_pos, init_radii = _init_grid()
        res = _global_search_de(init_pos, init_radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 3. Local refinement on best solution
    if best_pos is not None and time.time() < deadline - 5:
        refined_pos, refined_radii = _local_refine_so(best_pos, best_radii, deadline)
        s = np.sum(refined_radii)
        if s > best_score:
            best_score = s
            best_pos = refined_pos
            best_radii = refined_radii

    # 4. Additional refinement passes
    if best_pos is not None and time.time() < deadline - 3:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            refined_pos, refined_radii = _local_refine_so(best_pos, best_radii, deadline)
            s = np.sum(refined_radii)
            if s > best_score:
                best_score = s
                best_pos = refined_pos
                best_radii = refined_radii

    if best_pos is None:
        pos, radii = _init_grid()
        pos, radii = _post_process(pos, radii)
        return np.column_stack([pos, radii])

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
