"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta

def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    boundary_dists = 0.5 - np.abs(c)
    
    if n < 2:
        return -np.min(boundary_dists)
    
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    
    if r_soft <= 0:
        return 1.0
    return -r_soft

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _get_active_pairs(c, r, tol_factor=1.05):
    """Get active pair indices where dist < 2*r*tol_factor."""
    dist_matrix = squareform(pdist(c, metric='euclidean'))
    threshold = 2.0 * r * tol_factor
    i_idx, j_idx = np.where(dist_matrix < threshold)
    mask = i_idx < j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    return list(zip(i_idx.tolist(), j_idx.tolist()))

def _get_active_boundary(c, r, tol_factor=1.05):
    """Get active boundary indices."""
    boundary_dists = 0.5 - np.abs(c)
    min_boundary_per_point = np.min(boundary_dists, axis=1)
    boundary_threshold = r * tol_factor
    b_idx = np.where(min_boundary_per_point < boundary_threshold)[0]
    return list(b_idx.tolist())

def _penalty_objective(params, c_ref, r_target, mu):
    """
    Penalty-based objective: minimize -r + mu * (violation)^2
    where r is the minimum of all distances/2 and boundary distances.
    This is a smooth surrogate that encourages increasing r while penalizing violations.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        r = min_boundary
        # No pair violations
        return -r
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    r = min(min_pair_dist / 2.0, min_boundary)
    
    # Penalty for any constraint violation (dist < 2*r_target or boundary < r_target)
    pair_violations = np.maximum(0, 2.0 * r_target - dists)
    boundary_violations = np.maximum(0, r_target - boundary_dists.ravel())
    
    penalty = mu * (np.sum(pair_violations ** 2) + np.sum(boundary_violations ** 2))
    
    return -r + penalty

def _slsqp_polish(c0, max_time=20.0):
    """
    Polish using SLSQP with active constraints, but with a limit on constraint count.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    best_c = c.copy()
    best_r = r
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    max_outer_iters = 20
    max_constraints = 80  # Limit to keep SLSQP fast
    
    for k in range(max_outer_iters):
        if time.time() > start_time + max_time - 2.0:
            break
        
        pair_indices = _get_active_pairs(c, r, tol_factor=1.03)
        boundary_indices = _get_active_boundary(c, r, tol_factor=1.03)
        
        # If too many constraints, take only the closest ones
        if len(pair_indices) > max_constraints // 2:
            # Sort by distance and take closest
            dist_matrix = squareform(pdist(c, metric='euclidean'))
            pair_dists = [(dist_matrix[i, j], i, j) for i, j in pair_indices]
            pair_dists.sort(key=lambda x: x[0])
            pair_indices = [(i, j) for _, i, j in pair_dists[:max_constraints // 2]]
        
        if len(boundary_indices) > max_constraints // 2:
            boundary_dists = 0.5 - np.abs(c)
            min_b = np.min(boundary_dists, axis=1)
            boundary_indices = list(np.argsort(min_b)[:max_constraints // 2])
        
        if len(pair_indices) == 0 and len(boundary_indices) == 0:
            pair_indices = _get_active_pairs(c, r, tol_factor=1.1)
            boundary_indices = _get_active_boundary(c, r, tol_factor=1.1)
            if len(pair_indices) == 0 and len(boundary_indices) == 0:
                break

        def objective(params):
            return -params[-1]
        
        def objective_grad(params):
            grad = np.zeros_like(params)
            grad[-1] = -1.0
            return grad
            
        constraints = []
        
        for i, j in pair_indices:
            def make_pair_con(i, j):
                def con(params):
                    ci = params[2*i:2*i+2]
                    cj = params[2*j:2*j+2]
                    r_val = params[-1]
                    dist = np.linalg.norm(ci - cj)
                    return dist - 2.0 * r_val
                
                def con_grad(params):
                    ci = params[2*i:2*i+2]
                    cj = params[2*j:2*j+2]
                    diff = ci - cj
                    dist = np.linalg.norm(diff)
                    if dist < 1e-12:
                        grad_c = np.zeros(2*N+1)
                        grad_c[-1] = -2.0
                        return grad_c
                    grad_c = np.zeros(2*N+1)
                    grad_c[2*i:2*i+2] = diff / dist
                    grad_c[2*j:2*j+2] = -diff / dist
                    grad_c[-1] = -2.0
                    return grad_c
                
                return {'type': 'ineq', 'fun': con, 'jac': con_grad}
            
            constraints.append(make_pair_con(i, j))
        
        for k_idx in boundary_indices:
            x, y = c[k_idx]
            dist_left = x + 0.5
            dist_right = 0.5 - x
            dist_bottom = y + 0.5
            dist_top = 0.5 - y
            tol = r * 1.03
            
            if dist_left < tol:
                def make_left_con(k_idx):
                    def con(params):
                        return 0.5 + params[2*k_idx] - params[-1]
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k_idx] = 1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_left_con(k_idx))
                
            if dist_right < tol:
                def make_right_con(k_idx):
                    def con(params):
                        return 0.5 - params[2*k_idx] - params[-1]
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k_idx] = -1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_right_con(k_idx))
                
            if dist_bottom < tol:
                def make_bottom_con(k_idx):
                    def con(params):
                        return 0.5 + params[2*k_idx+1] - params[-1]
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k_idx+1] = 1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_bottom_con(k_idx))
                
            if dist_top < tol:
                def make_top_con(k_idx):
                    def con(params):
                        return 0.5 - params[2*k_idx+1] - params[-1]
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k_idx+1] = -1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_top_con(k_idx))
        
        if not constraints:
            break
            
        x0 = np.zeros(2*N + 1)
        x0[:2*N] = c.flatten()
        x0[-1] = r
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 120, 'ftol': 1e-15, 'disp': False}
            )
            
            new_c = result.x[:2*N].reshape(-1, 2)
            new_c = np.clip(new_c, -0.5, 0.5)
            new_r = _compute_radius(new_c)
            
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
                c = new_c.copy()
                r = new_r
            else:
                break
                
        except Exception:
            break
            
    return best_c, best_r


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with adaptive beta."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    betas = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
            
    return best_c, best_r


def _jitter_and_reoptimize(c0, r0, max_time=20.0, n_trials=8):
    """
    Try small random perturbations and re-optimize to escape local minima.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    
    rng = np.random.RandomState(42)
    
    for trial in range(n_trials):
        if time.time() > start_time + max_time - 5:
            break
        
        # Small random perturbation
        scale = 0.008 * (1.0 - trial / n_trials)  # Decreasing scale
        c_perturbed = best_c + rng.randn(N, 2) * scale
        c_perturbed = np.clip(c_perturbed, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Quick smooth optimization from perturbed start
        c_opt, r_opt = _local_optimize_smooth(c_perturbed, max_time=10.0)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    # 1. Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
        
    # 2. Smooth Optimization (Warm-up) - use most of the time here
    if time.time() < deadline - 50:
        remaining = deadline - time.time() - 50
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(45.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
            
    # 3. Jitter and re-optimize to escape local minima
    if best_c is not None and time.time() < deadline - 30:
        remaining = min(25.0, deadline - time.time() - 30)
        c_jitter, r_jitter = _jitter_and_reoptimize(best_c, best_radius, max_time=remaining)
        if r_jitter > best_radius:
            best_radius = r_jitter
            best_c = c_jitter.copy()
            
    # 4. SLSQP Polish (High Precision)
    if best_c is not None and time.time() < deadline - 15:
        remaining = min(25.0, deadline - time.time() - 15)
        c_polished, r_polished = _slsqp_polish(best_c, max_time=remaining)
        if r_polished > best_radius:
            best_radius = r_polished
            best_c = c_polished.copy()
            
    # 5. Final SLSQP pass with tighter tolerance
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(3.0, deadline - time.time() - 3)
        c_final, r_final = _slsqp_polish(best_c, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    if best_c is None:
        best_c = _initialize_hexagonal()
        
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
