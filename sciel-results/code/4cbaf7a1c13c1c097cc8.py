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
    """
    Attempt to solve using a smooth penalty-based gradient method (L-BFGS-B).
    This approximates the SOCP-like geometric refinement by minimizing a penalty
    function of constraint violations, which is smooth and differentiable.
    """
    from scipy.optimize import minimize
    
    def penalty_obj(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        
        # Boundary penalties: (r - x)^2 + (r - (1-x))^2 if violation
        # We use max(0, violation)^2
        viol_x_left = np.maximum(0, r - p[:, 0])
        viol_x_right = np.maximum(0, r - (1.0 - p[:, 0]))
        viol_y_bottom = np.maximum(0, r - p[:, 1])
        viol_y_top = np.maximum(0, r - (1.0 - p[:, 1]))
        
        bound_pen = np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + \
                    np.sum(viol_y_bottom**2) + np.sum(viol_y_top**2)
        
        # Overlap penalties: (r_i + r_j - dist)^2 if violation
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        diff = min_dist - dist
        # We only care about upper triangle
        # To avoid sqrt(0) issues in gradient, we add a small epsilon
        # Actually, L-BFGS-B handles non-smoothness poorly, but max(0, x)^2 is C1
        # dist is smooth if dist > 0.
        
        # Flatten the masked array for sum
        # Note: diff is (N, N), we take upper triangle
        # We can use np.where(mask, diff, 0)
        viol = np.where(mask, diff, 0.0)
        viol = np.maximum(0, viol)
        overlap_pen = np.sum(viol**2)
        
        # Objective: Maximize sum(r) -> Minimize -sum(r) + lambda * penalties
        # Lambda needs to be large enough to enforce constraints
        lam = 100.0
        return -np.sum(r) + lam * (bound_pen + overlap_pen)
        
    # We need an initial guess for this function to be called correctly?
    # The current signature is _solve_cvxpy(deadline). It doesn't take pos/radii.
    # This is a problem. The original _solve_cvxpy was a black box.
    # I need to change how it's called or change the signature.
    # But I must keep signatures? "Keep the function names and signatures of the seed program."
    # _solve_cvxpy(deadline) is the signature.
    # I cannot pass initial pos/radii.
    # I will use a default grid init inside if not provided?
    # Or I can modify construct to pass it? No, construct calls _solve_cvxpy(deadline).
    # I will generate a random init inside _solve_cvxpy.
    
    # However, _local_refine also calls _solve_slsqp_vectorized which takes pos/radii.
    # I should probably create a new function _solve_smooth(pos, radii, deadline)
    # And modify _solve_cvxpy to just return None or use a default init.
    # Better: I will replace _solve_cvxpy with a wrapper that does nothing or uses default,
    # and add _solve_smooth.
    
    # Actually, I can just return None here and rely on the other methods,
    # OR I can change the internal logic of construct to call a new function.
    # Let's replace _solve_cvxpy with a function that uses a default grid init.
    
    pos, radii = _init_grid()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    try:
        res = minimize(penalty_obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.success:
            p = res.x[:2*N].reshape(N, 2)
            r = res.x[2*N:]
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                return p, r
    except Exception:
        pass
    return None

def _solve_smooth(pos, radii, deadline):
    """Solve using L-BFGS-B with penalty function (smooth, fast)."""
    from scipy.optimize import minimize
    
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    def penalty_obj(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        
        viol_x_left = np.maximum(0, r - p[:, 0])
        viol_x_right = np.maximum(0, r - (1.0 - p[:, 0]))
        viol_y_bottom = np.maximum(0, r - p[:, 1])
        viol_y_top = np.maximum(0, r - (1.0 - p[:, 1]))
        
        bound_pen = np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + \
                    np.sum(viol_y_bottom**2) + np.sum(viol_y_top**2)
        
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        diff = min_dist - dist
        viol = np.where(mask, diff, 0.0)
        viol = np.maximum(0, viol)
        overlap_pen = np.sum(viol**2)
        
        lam = 100.0
        return -np.sum(r) + lam * (bound_pen + overlap_pen)
        
    try:
        res = minimize(penalty_obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-12})
        if res.success:
            p = res.x[:2*N].reshape(N, 2)
            r = res.x[2*N:]
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                return p, r
    except Exception:
        pass
    return None

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
            
            # First try fast smooth solver
            res = _solve_smooth(perturbed_pos, perturbed_radii, deadline)
            if res:
                p, r = res
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_pos = p.copy()
                    best_radii = r.copy()
            
            # Then try SLSQP for polishing
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
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        pos, radii = _init_random(seed)
        # Use smooth solver first for speed
        res = _solve_smooth(pos, radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r
        # Then SLSQP
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
