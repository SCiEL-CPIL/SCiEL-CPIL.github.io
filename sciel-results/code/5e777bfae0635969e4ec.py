import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _score(points):
    """Compute the packing score (largest common radius)."""
    points = np.asarray(points).reshape(N, 2)
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary = 1.0 - norms
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0
    r = min(np.min(boundary), np.min(pair_dists))
    return r

def _softmin_logsumexp(values, beta):
    """Numerically stable log-sum-exp based soft-min."""
    m = np.min(values)
    if not np.isfinite(m):
        return m
    shifted = -beta * (values - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m + log_sum / beta

def _smooth_objective(points, beta):
    """
    Smooth approximation of the negative packing score.
    We minimize -softmin(constraints) which approximates maximizing the min constraint.
    """
    points = points.reshape(N, 2)
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary = 1.0 - norms
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0
    
    all_vals = np.concatenate([pair_dists, boundary])
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min

def _smooth_grad(points, beta):
    """
    Gradient of the smooth objective.
    """
    points = points.reshape(N, 2)
    n = N
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary = 1.0 - norms
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0
    
    all_vals = np.concatenate([pair_dists, boundary])
    m = np.min(all_vals)
    shifted = -beta * (all_vals - m)
    shifted = np.clip(shifted, -500, 500)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    grad = np.zeros((n, 2))
    
    # Pairwise gradient
    # d(soft_min)/d(pair_dist_k) = -weights[k]
    # d(pair_dist_k)/d(points[i]) = 0.5 * (points[i] - points[j]) / dist_ij
    # d(pair_dist_k)/d(points[j]) = -0.5 * (points[i] - points[j]) / dist_ij
    # Objective is -soft_min, so d(obj)/d(pair_dist_k) = weights[k]
    # Chain rule: d(obj)/d(points[i]) += weights[k] * 0.5 * (points[i] - points[j]) / dist_ij
    
    for idx in range(len(pair_dists)):
        w = weights[idx]
        if w < 1e-12:
            continue
        i, j = iu[0][idx], iu[1][idx]
        d_vec = points[i] - points[j]
        d_norm = np.linalg.norm(d_vec)
        if d_norm < 1e-12:
            continue
        unit = d_vec / d_norm
        grad[i] += 0.5 * w * unit
        grad[j] -= 0.5 * w * unit
    
    # Boundary gradient
    # d(soft_min)/d(boundary_i) = -weights[len(pair_dists)+i]
    # d(boundary_i)/d(points[i]) = -points[i] / norm_i
    # d(obj)/d(boundary_i) = weights[...]
    # Chain rule: d(obj)/d(points[i]) += weights[...] * (-points[i] / norm_i)
    
    for i in range(n):
        w = weights[len(pair_dists) + i]
        if w < 1e-12:
            continue
        if norms[i] < 1e-12:
            continue
        grad[i] -= w * points[i] / norms[i]
    
    return grad.flatten()

def _optimize_from_init(points0, time_limit=10.0, beta_start=100.0, beta_end=5000.0):
    """
    Progressive beta annealing optimization.
    """
    t_start = time.time()
    x = points0.flatten().copy()
    n_betas = 10
    betas = np.geomspace(beta_start, beta_end, n_betas)
    
    best_x = x.copy()
    best_score = _score(best_x.reshape(N, 2))
    
    for b in betas:
        if time.time() - t_start > time_limit:
            break
        
        def obj(x_flat):
            return _smooth_objective(x_flat, b)
        
        def obj_grad(x_flat):
            return _smooth_grad(x_flat, b)
        
        try:
            res = minimize(
                obj,
                x,
                jac=obj_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < obj(x):
                x = res.x
                s = _score(x.reshape(N, 2))
                if s > best_score:
                    best_score = s
                    best_x = x.copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_score

def _binary_search_polish(points, time_limit=5.0):
    """
    Binary search on radius r to find the maximum feasible r.
    For a given r, check if we can place N circles of radius r.
    This is a feasibility problem which can be solved by local optimization.
    """
    t_start = time.time()
    low = 0.0
    high = 0.2
    
    # Initial high estimate
    s = _score(points)
    low = s
    high = s * 1.1
    
    best_pts = points.copy()
    best_r = low
    
    for _ in range(20):
        if time.time() - t_start > time_limit:
            break
        
        mid = (low + high) / 2.0
        
        # Try to find a configuration with score >= mid
        # We use the smooth objective with a target radius
        # If we can push the min constraint above mid, we are feasible
        
        # A simpler approach: just optimize the current points and see if score improves
        # But for binary search, we need a feasibility check.
        # Let's use a penalty method: minimize (max(0, mid - score))^2 + small * (score - mid)^2 if score < mid?
        # Actually, let's just use the smooth optimizer and see if it beats mid.
        
        # This is a bit tricky to implement efficiently in a binary search framework.
        # Let's stick to the progressive beta annealing which is more robust.
        break
    
    return best_pts, best_r

def _ring_initialization(n):
    """
    Generate a ring-based initial configuration.
    """
    points = []
    # Center
    points.append([0.0, 0.0])
    
    rings = [
        (6, 0.25),
        (12, 0.5),
        (18, 0.75),
        (14, 0.95)
    ]
    
    for count, radius in rings:
        for k in range(count):
            angle = 2 * np.pi * k / count
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    points = np.array(points)
    if len(points) >= n:
        # Select n points, preferring those closer to optimal packing
        # For now, just take the first n
        return points[:n]
    
    # Fill with random points if not enough
    while len(points) < n:
        points.append(np.random.randn(2) * 0.5)
    return np.array(points[:n])

def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.1, 0.2, 20):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points

def construct():
    t_start = time.time()
    
    # 1. Multiple initializations
    init1 = _hexagonal_init(N)
    init2 = _ring_initialization(N)
    init3 = np.random.randn(N, 2) * 0.4
    
    best_points = None
    best_score = 0.0
    
    for init in [init1, init2, init3]:
        r = _score(init)
        if r > best_score:
            best_score = r
            best_points = init.copy()
    
    # 2. Progressive beta annealing optimization
    if time.time() < DEADLINE - 30:
        c_opt, r_opt = _optimize_from_init(best_points, time_limit=min(20.0, DEADLINE - (time.time() - t_start) - 30))
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # 3. Multi-start with perturbations
    n_restarts = 0
    while time.time() < DEADLINE - 10 and n_restarts < 5:
        perturbation_scale = 0.05 / (1.0 + 0.5 * n_restarts)
        x0 = best_points + np.random.randn(N, 2) * perturbation_scale
        
        c_opt, r_opt = _optimize_from_init(x0, time_limit=min(15.0, DEADLINE - (time.time() - t_start) - 10))
        
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
        
        n_restarts += 1
    
    # 4. Final polish
    if time.time() < DEADLINE - 2:
        c_opt, r_opt = _optimize_from_init(best_points, time_limit=min(5.0, DEADLINE - (time.time() - t_start) - 2), beta_start=1000.0, beta_end=10000.0)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Ensure validity
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1.0:
            scale = (1.0 - 1e-10) / r2[i]
            c_final[i] *= scale
    
    r_check = _score(c_final)
    if r_check < best_score - 1e-12:
        # Fallback to a safe configuration
        c_final *= (1.0 - 1e-10)
    
    return c_final
