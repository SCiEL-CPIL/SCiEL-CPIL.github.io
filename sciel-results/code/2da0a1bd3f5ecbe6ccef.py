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
    
    for _ in range(500):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-8)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-15) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-8)
                        radii[j] = max(radii[j] * k, 1e-8)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _solve_socp_cvxpy():
    """Solve the SOCP formulation directly using CVXPY."""
    try:
        import cvxpy as cp
    except ImportError:
        return None
    
    # Variables
    x = cp.Variable((N, 2))
    r = cp.Variable(N)
    
    # Objective: maximize sum of radii
    objective = cp.Maximize(cp.sum(r))
    
    # Constraints
    constraints = [
        r >= 0,
        x[:, 0] >= r,
        1 - x[:, 0] >= r,
        x[:, 1] >= r,
        1 - x[:, 1] >= r,
    ]
    
    # Non-overlap constraints: ||x_i - x_j|| >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append(cp.norm(x[i] - x[j], 2) >= r[i] + r[j])
    
    prob = cp.Problem(objective, constraints)
    
    # Try multiple solvers in order of preference
    solvers = []
    try:
        if cp.CLARABEL is not None:
            solvers.append(cp.CLARABEL)
    except:
        pass
    try:
        if cp.SCS is not None:
            solvers.append(cp.SCS)
    except:
        pass
    try:
        if cp.ECOS is not None:
            solvers.append(cp.ECOS)
    except:
        pass
    
    if not solvers:
        return None
    
    for solver in solvers:
        try:
            prob.solve(solver=solver)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                pos = np.array(x.value)
                radii = np.array(r.value)
                pos, radii = _post_process(pos, radii)
                if _is_valid(pos, radii):
                    return pos, radii
        except Exception:
            continue
    
    return None

def _solve_socp_scipy():
    """Solve using scipy's trust-constr with careful formulation."""
    from scipy.optimize import minimize
    
    # Initial guess: uniform grid
    pos = np.zeros((N, 2))
    k = 0
    for i in range(5):
        for j in range(5):
            if k >= N:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            k += 1
    radii = np.full(N, 0.09)
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Bounds
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-8, 0.5)] * N
    
    def objective(x):
        return -np.sum(x[2*N:])
    
    def constraints_fun(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        
        # Boundary constraints (must be >= 0)
        c1 = p[:, 0] - r
        c2 = 1.0 - p[:, 0] - r
        c3 = p[:, 1] - r
        c4 = 1.0 - p[:, 1] - r
        
        # Non-overlap: dist - (r_i + r_j) >= 0
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        c5 = (dist - min_dist)[mask]
        
        return np.concatenate([c1, c2, c3, c4, c5])
    
    try:
        res = minimize(objective, x0, method='trust-constr', bounds=bounds,
                      constraints=[{'type': 'ineq', 'fun': constraints_fun}],
                      options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-8})
        if res.success or res.status == 1:
            p = res.x[:2*N].reshape(N, 2)
            r = res.x[2*N:]
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                return p, r
    except Exception:
        pass
    
    return None

def _local_refine_socp(pos, radii, deadline):
    """Refine using SOCP from multiple starting points."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(123)
    
    # Try perturbing the current best and re-solving
    for attempt in range(10):
        if time.time() > deadline - 3:
            break
        
        # Small perturbation
        scale = 0.02 * (1 - attempt / 10.0)
        perturbed_pos = best_pos + rng.normal(0, scale, (N, 2))
        perturbed_pos = np.clip(perturbed_pos, 0.01, 0.99)
        perturbed_radii = best_radii.copy()
        
        perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
        
        # Try to improve via SLSQP from this point
        from scipy.optimize import minimize
        
        x0 = np.concatenate([perturbed_pos.flatten(), perturbed_radii])
        bounds = [(0.0, 1.0)] * (2 * N) + [(1e-8, 0.5)] * N
        
        def obj(x):
            return -np.sum(x[2*N:])
            
        def cons(x):
            p = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            c_bound = np.concatenate([
                p[:, 0] - r,
                1.0 - p[:, 0] - r,
                p[:, 1] - r,
                1.0 - p[:, 1] - r
            ])
            dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
            dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[:, np.newaxis] + r[np.newaxis, :]
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            c_overlap = (dist - min_dist)[mask]
            return np.concatenate([c_bound, c_overlap])
        
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                          constraints=[{'type': 'ineq', 'fun': cons}],
                          options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                p = res.x[:2*N].reshape(N, 2)
                r = res.x[2*N:]
                p, r = _post_process(p, r)
                if _is_valid(p, r):
                    s = np.sum(r)
                    if s > best_score:
                        best_score = s
                        best_pos = p.copy()
                        best_radii = r.copy()
        except Exception:
            pass
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_pos = None
    best_radii = None
    best_score = -1
    
    # 1. Primary: Solve SOCP directly via CVXPY
    res = _solve_socp_cvxpy()
    if res:
        pos, radii = res
        score = np.sum(radii)
        best_score = score
        best_pos = pos
        best_radii = radii
    
    # 2. Fallback: Solve via scipy trust-constr
    if best_pos is None and time.time() < deadline - 10:
        res = _solve_socp_scipy()
        if res:
            pos, radii = res
            score = np.sum(radii)
            best_score = score
            best_pos = pos
            best_radii = radii
    
    # 3. If still no solution, use grid init + SLSQP
    if best_pos is None and time.time() < deadline - 10:
        pos = np.zeros((N, 2))
        k = 0
        for i in range(5):
            for j in range(5):
                if k >= N:
                    break
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                k += 1
        radii = np.full(N, 0.09)
        pos, radii = _post_process(pos, radii)
        
        from scipy.optimize import minimize
        x0 = np.concatenate([pos.flatten(), radii])
        bounds = [(0.0, 1.0)] * (2 * N) + [(1e-8, 0.5)] * N
        
        def obj(x):
            return -np.sum(x[2*N:])
            
        def cons(x):
            p = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            c_bound = np.concatenate([
                p[:, 0] - r,
                1.0 - p[:, 0] - r,
                p[:, 1] - r,
                1.0 - p[:, 1] - r
            ])
            dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
            dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[:, np.newaxis] + r[np.newaxis, :]
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            c_overlap = (dist - min_dist)[mask]
            return np.concatenate([c_bound, c_overlap])
        
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                          constraints=[{'type': 'ineq', 'fun': cons}],
                          options={'maxiter': 300, 'ftol': 1e-12})
            if res.success:
                p = res.x[:2*N].reshape(N, 2)
                r = res.x[2*N:]
                p, r = _post_process(p, r)
                if _is_valid(p, r):
                    best_score = np.sum(r)
                    best_pos = p
                    best_radii = r
        except Exception:
            pass
    
    # 4. Local refinement
    if best_pos is not None and time.time() < deadline - 5:
        refined_pos, refined_radii = _local_refine_socp(best_pos, best_radii, deadline)
        s = np.sum(refined_radii)
        if s > best_score:
            best_score = s
            best_pos = refined_pos
            best_radii = refined_radii
    
    # Final fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        k = 0
        for i in range(5):
            for j in range(5):
                if k >= N:
                    break
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                k += 1
        radii = np.full(N, 0.09)
        pos, radii = _post_process(pos, radii)
        return np.column_stack([pos, radii])
    
    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
