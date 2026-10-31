# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

def _compute_score(pos, radii):
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
    
    for _ in range(200):
        valid = True
        
        # Boundary constraints
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-6)
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
                        k = dist / sum_r * 0.99999
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

def _init_structured():
    """Structured initialization based on known good packings."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a 5x5 grid pattern with some perturbation
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx >= N:
                break
            x = (j + 0.5) / 5.0
            y = (i + 0.5) / 5.0
            pos[idx] = [x, y]
            radii[idx] = 0.085
    
    # Add small random perturbations to break symmetry
    rng = np.random.RandomState(42)
    pos += rng.normal(0, 0.01, (N, 2))
    pos = np.clip(pos, 0.05, 0.95)
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.06, 0.10, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _solve_slsqp(pos, radii, deadline):
    """Solve using SLSQP with efficient constraint evaluation."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = [(0.02, 0.98)] * (2 * N) + [(0.02, 0.22)] * N
    
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
        
        # Overlap constraints - only compute upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        i_idx, j_idx = np.where(mask)
        dx = p[i_idx, 0] - p[j_idx, 0]
        dy = p[i_idx, 1] - p[j_idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        c_overlap = dist - (r[i_idx] + r[j_idx])
        
        return np.concatenate([c_bound, c_overlap])
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': cons_fun}],
                       options={'maxiter': 300, 'ftol': 1e-10})
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

def _solve_cvxpy(pos_init, radii_init, deadline):
    """Attempt to solve using CVXPY SOCP formulation."""
    try:
        import cvxpy as cp
    except ImportError:
        return None
        
    x = cp.Variable((N, 2))
    r = cp.Variable(N)
    
    objective = cp.Maximize(cp.sum(r))
    
    constraints = [
        r >= 0.01,
        x[:, 0] - r >= 0,
        1 - x[:, 0] - r >= 0,
        x[:, 1] - r >= 0,
        1 - x[:, 1] - r >= 0,
    ]
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append(cp.norm(x[i] - x[j], 2) >= r[i] + r[j])
            
    prob = cp.Problem(objective, constraints)
    
    # Set initial values for warm start
    try:
        x.value = pos_init
        r.value = radii_init
    except:
        pass
    
    try:
        solvers_to_try = []
        try:
            solvers_to_try.append(cp.CLARABEL)
        except:
            pass
        try:
            solvers_to_try.append(cp.SCS)
        except:
            pass
        try:
            solvers_to_try.append(cp.ECOS)
        except:
            pass
            
        success = False
        for solver in solvers_to_try:
            try:
                prob.solve(solver=solver, warm_start=True, max_iters=500)
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    success = True
                    break
            except Exception:
                continue
        
        if not success:
            return None
    except Exception:
        return None
        
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None
        
    pos = np.array(x.value)
    radii = np.array(r.value)
    
    radii = np.maximum(radii, 1e-6)
    
    pos, radii = _post_process(pos, radii)
    if not _is_valid(pos, radii):
        return None
        
    return pos, radii

def _local_refine(pos, radii, deadline):
    """Iterative local refinement with multiple strategies."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(123)
    
    # Strategy 1: Small perturbations + SLSQP
    while time.time() < deadline - 5:
        perturbed_pos = best_pos + rng.normal(0, 0.01, (N, 2))
        perturbed_pos = np.clip(perturbed_pos, 0.03, 0.97)
        perturbed_radii = best_radii * (1 + rng.normal(0, 0.05, N))
        perturbed_radii = np.clip(perturbed_radii, 0.02, 0.20)
        
        perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
        
        res = _solve_slsqp(perturbed_pos, perturbed_radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p.copy()
                best_radii = r.copy()
    
    # Strategy 2: Larger perturbations to escape local optima
    while time.time() < deadline - 3:
        perturbed_pos = best_pos + rng.normal(0, 0.03, (N, 2))
        perturbed_pos = np.clip(perturbed_pos, 0.05, 0.95)
        perturbed_radii = best_radii * (1 + rng.normal(0, 0.1, N))
        perturbed_radii = np.clip(perturbed_radii, 0.03, 0.18)
        
        perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
        
        res = _solve_slsqp(perturbed_pos, perturbed_radii, deadline)
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
    deadline = start_time + 115.0

    best_pos = None
    best_radii = None
    best_score = -1

    # 1. Try structured initialization with SLSQP
    pos, radii = _init_structured()
    res = _solve_slsqp(pos, radii, deadline)
    if res:
        p, r = res
        s = np.sum(r)
        if s > best_score:
            best_score = s
            best_pos = p
            best_radii = r

    # 2. Try multiple random initializations
    for seed in range(10):
        if time.time() > deadline - 15:
            break
        pos, radii = _init_random(seed)
        res = _solve_slsqp(pos, radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 3. Try SOCP via CVXPY with best solution as warm start
    if best_pos is not None and time.time() < deadline - 10:
        res = _solve_cvxpy(best_pos, best_radii, deadline)
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
        pos, radii = _init_structured()
        pos, radii = _post_process(pos, radii)
        return np.column_stack([pos, radii])

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
