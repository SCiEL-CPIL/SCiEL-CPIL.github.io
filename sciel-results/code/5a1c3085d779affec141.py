# EVOLVE-BLOCK-START

import numpy as np
import time

N = 26

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

def _init_grid():
    """Grid-based initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k >= N:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[k] = 0.095
            k += 1
    if k < N:
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _solve_cvxpy(deadline):
    """Attempt to solve using CVXPY SOCP formulation."""
    try:
        import cvxpy as cp
    except ImportError:
        return None
        
    x = cp.Variable((N, 2))
    r = cp.Variable(N)
    
    objective = cp.Maximize(cp.sum(r))
    
    constraints = [
        r >= 0,
        x[:, 0] - r >= 0,
        1 - x[:, 0] - r >= 0,
        x[:, 1] - r >= 0,
        1 - x[:, 1] - r >= 0,
    ]
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append(cp.norm(x[i] - x[j], 2) >= r[i] + r[j])
            
    prob = cp.Problem(objective, constraints)
    
    try:
        # Try SCS first, then Clarabel
        prob.solve(solver=cp.SCS, warm_start=True)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            prob.solve(solver=cp.CLARABEL)
    except Exception:
        return None
        
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None
        
    pos = np.array(x.value)
    radii = np.array(r.value)
    
    pos, radii = _post_process(pos, radii)
    if not _is_valid(pos, radii):
        return None
        
    return pos, radii

def _solve_slsqp_vectorized(pos, radii, deadline):
    """Solve using SLSQP with vectorized constraints (faster than lambda list)."""
    from scipy.optimize import minimize
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    def obj(x):
        return -np.sum(x[2*N:])
        
    def cons_fun(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        # Boundary constraints
        c_bound = np.concatenate([
            p[:, 0] - r,
            1.0 - p[:, 0] - r,
            p[:, 1] - r,
            1.0 - p[:, 1] - r
        ])
        
        # Overlap constraints
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        c_overlap = (dist - min_dist)[mask]
        
        return np.concatenate([c_bound, c_overlap])
        
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': cons_fun}],
                       options={'maxiter': 300, 'ftol': 1e-10})
        if res.success:
            p = res.x[:2*N].reshape(N, 2)
            r = res.x[2*N:]
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                return p, r
    except Exception:
        pass
    return None

def _local_refine(pos, radii, deadline):
    """Iterative local refinement with random perturbations."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 2:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            
            perturbed_pos = best_pos + rng.normal(0, 0.01, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.05, 0.95)
            perturbed_radii = best_radii.copy()
            
            perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
            
            res = _solve_slsqp_vectorized(perturbed_pos, perturbed_radii, deadline)
            if res:
                p, r = res
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_pos = p.copy()
                    best_radii = r.copy()
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    best_pos = None
    best_radii = None
    best_score = -1

    # 1. Try SOCP via CVXPY
    if time.time() < deadline - 5:
        res = _solve_cvxpy(deadline)
        if res:
            pos, radii = res
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos
                best_radii = radii

    # 2. Fallback: Local Search with Vectorized SLSQP
    if time.time() < deadline - 10:
        # Grid Init
        pos, radii = _init_grid()
        res = _solve_slsqp_vectorized(pos, radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 3. More Random Inits if time permits
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        pos, radii = _init_random(seed)
        res = _solve_slsqp_vectorized(pos, radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 4. Local refinement on best solution
    if best_pos is not None and time.time() < deadline - 5:
        refined_pos, refined_radii = _local_refine(best_pos, best_radii, deadline)
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
