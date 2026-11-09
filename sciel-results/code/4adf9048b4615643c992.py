# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial.distance as spd
import scipy.optimize as opt
import time

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _normalize_points(points):
    """Normalize points: center at origin, scale so sum of squared norms = 1."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    norm = np.sqrt(np.sum(centered**2))
    if norm < 1e-12:
        return centered + 1e-6 * np.random.randn(*centered.shape)
    return centered / norm

def _soft_log_min_max(points_flat, beta=0.05):
    """
    Smooth surrogate: approximate -log(d_min) + log(d_max) using log-sum-exp.
    d_min approx: log-sum-exp of -dists/beta (gives ~ -log(min(dist)))
    d_max approx: log-sum-exp of dists/beta (gives ~ log(max(dist)))
    """
    n = len(points_flat) // 2
    points = points_flat.reshape(n, 2)
    points = _normalize_points(points)
    
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 1e6
    
    # Soft min: -log(min(dists)) ≈ log-sum-exp(-dists/beta) - log(count)
    # Actually: min(dists) ≈ exp(-logsumexp(-dists/beta) * beta) / beta... 
    # Let's use: -log(min(dists)) ≈ logsumexp(-dists/beta) * beta / ... 
    # Simpler: use logsumexp directly
    # -log(min(d)) = logsumexp(-d/beta) * beta - log(K) approximately for small beta
    # Actually logsumexp(-d/beta) ≈ -min(d)/beta + log(K) for small beta
    # So -min(d)/beta ≈ logsumexp(-d/beta) - log(K)
    # min(d) ≈ -beta * (logsumexp(-d/beta) - log(K))
    # -log(min(d)) ≈ -log(-beta * (logsumexp(-d/beta) - log(K)))
    
    K = len(dists)
    # For soft min of dists:
    # soft_min = -beta * (logsumexp(-dists/beta) - log(K))
    neg_dists_over_beta = -dists / beta
    max_neg = np.max(neg_dists_over_beta)
    logsumexp_neg = max_neg + np.log(np.sum(np.exp(neg_dists_over_beta - max_neg)))
    soft_min_val = -beta * (logsumexp_neg - np.log(K))
    
    # For soft max of dists:
    # soft_max = beta * (logsumexp(dists/beta) - log(K))
    dists_over_beta = dists / beta
    max_pos = np.max(dists_over_beta)
    logsumexp_pos = max_pos + np.log(np.sum(np.exp(dists_over_beta - max_pos)))
    soft_max_val = beta * (logsumexp_pos - np.log(K))
    
    if soft_min_val <= 0 or soft_max_val <= 0:
        return 1e6
    
    return -np.log(soft_min_val) + np.log(soft_max_val)

def _soft_log_min_max_grad(points_flat, beta=0.05):
    """
    Analytic gradient of the smooth surrogate objective.
    """
    n = len(points_flat) // 2
    points = points_flat.reshape(n, 2).copy()
    points = _normalize_points(points)
    
    dists = spd.pdist(points)
    K = len(dists)
    
    if K == 0:
        return np.zeros_like(points_flat)
    
    # We need d/dpoints of [-log(soft_min) + log(soft_max)]
    # soft_min = -beta * (logsumexp(-dists/beta) - log(K))
    # soft_max = beta * (logsumexp(dists/beta) - log(K))
    
    # d(soft_min)/d(dist_i) = -beta * d(logsumexp(-dists/beta))/d(dist_i)
    # d(logsumexp(-dists/beta))/d(dist_i) = -exp(-dist_i/beta) / sum_j exp(-dist_j/beta)
    # = -p_i where p_i = exp(-dist_i/beta) / sum_j exp(-dist_j/beta)
    # So d(soft_min)/d(dist_i) = beta * p_i
    
    # d(soft_max)/d(dist_i) = beta * d(logsumexp(dists/beta))/d(dist_i)
    # = beta * exp(dist_i/beta) / sum_j exp(dist_j/beta)
    # = beta * q_i where q_i = exp(dist_i/beta) / sum_j exp(dist_j/beta)
    
    # Objective = -log(soft_min) + log(soft_max)
    # d(obj)/d(dist_i) = -1/soft_min * d(soft_min)/d(dist_i) + 1/soft_max * d(soft_max)/d(dist_i)
    # = -1/soft_min * beta * p_i + 1/soft_max * beta * q_i
    
    # Now d(dist_i)/d(points) needs the pair indices
    
    # Build pair index mapping for pdist
    # pdist returns distances in order: (0,1),(0,2),...,(0,n-1),(1,2),...
    pair_indices = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_indices.append((i, j, idx))
            idx += 1
    
    neg_dists_over_beta = -dists / beta
    max_neg = np.max(neg_dists_over_beta)
    exp_neg = np.exp(neg_dists_over_beta - max_neg)
    sum_exp_neg = np.sum(exp_neg)
    p = exp_neg / sum_exp_neg  # softmax weights for soft min
    
    dists_over_beta = dists / beta
    max_pos = np.max(dists_over_beta)
    exp_pos = np.exp(dists_over_beta - max_pos)
    sum_exp_pos = np.sum(exp_pos)
    q = exp_pos / sum_exp_pos  # softmax weights for soft max
    
    logsumexp_neg = max_neg + np.log(sum_exp_neg)
    logsumexp_pos = max_pos + np.log(sum_exp_pos)
    
    soft_min_val = -beta * (logsumexp_neg - np.log(K))
    soft_max_val = beta * (logsumexp_pos - np.log(K))
    
    if soft_min_val <= 0 or soft_max_val <= 0:
        return np.zeros_like(points_flat)
    
    # d(obj)/d(dist_k) for each pair k
    d_obj_ddist = -beta * p / soft_min_val + beta * q / soft_max_val
    
    # Now accumulate gradient w.r.t. points
    # dist_k = ||p_i - p_j||
    # d(dist_k)/d(p_i) = (p_i - p_j) / dist_k
    # d(dist_k)/d(p_j) = (p_j - p_i) / dist_k
    
    grad_points = np.zeros((n, 2))
    
    for (i, j, k) in pair_indices:
        diff = points[i] - points[j]
        dist_k = dists[k]
        if dist_k < 1e-12:
            continue
        d_dist_dp_i = diff / dist_k
        d_dist_dp_j = -diff / dist_k
        
        coeff = d_obj_ddist[k]
        grad_points[i] += coeff * d_dist_dp_i
        grad_points[j] += coeff * d_dist_dp_j
    
    # Now we need to account for the normalization step
    # points_norm = (points - centroid) / norm
    # This is a more complex chain rule. For simplicity, we'll use the 
    # gradient on the normalized space and note that the normalization is 
    # scale-invariant in the objective, so the gradient should be approximately 
    # valid in the original space too (up to a scale factor that doesn't affect 
    # the direction much for L-BFGS-B).
    
    # Actually, let's just return the gradient in the normalized coordinate system
    # and let the optimizer handle it. The key insight is that the objective is 
    # scale-invariant, so the gradient direction is what matters.
    
    return grad_points.flatten()

def _nelder_mead_polish(points, maxiter=200, tol=1e-8):
    """Polish with Nelder-Mead on the non-smooth objective."""
    def obj(x_flat):
        pts = _normalize_points(x_flat.reshape(-1, 2))
        dists = spd.pdist(pts)
        if len(dists) == 0:
            return 1e6
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 0 or d_max <= 0:
            return 1e6
        return -np.log(d_min) + np.log(d_max)
    
    try:
        res = opt.minimize(obj, points.flatten(), method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol})
        return res.x.reshape(-1, 2)
    except Exception:
        return points

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    def update_best(points):
        nonlocal best_score, best_points
        score = calculate_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        return score
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hexagonal grid 4x4
    rows, cols = 4, 4
    dx = 1.0
    dy = dx * np.sqrt(3) / 2
    hex_points = []
    for i in range(rows):
        for j in range(cols):
            x = j * dx + (i % 2) * (dx / 2)
            y = i * dy
            hex_points.append([x, y])
    inits.append(np.array(hex_points))
    
    # 2. 4x4 square grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    inits.append(np.c_[grid_x.flatten(), grid_y.flatten()])
    
    # 3. Two concentric rings (8+8)
    angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    for r_outer, r_inner in [(1.0, 0.5), (1.0, 0.6), (1.0, 0.4)]:
        circle_points = np.vstack([
            np.c_[r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)],
            np.c_[r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)]
        ])
        inits.append(circle_points)
    
    # 4. Three rings: 6+6+4
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    angles3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    ring3 = np.vstack([
        np.c_[1.0 * np.cos(angles1), 1.0 * np.sin(angles1)],
        np.c_[0.55 * np.cos(angles2), 0.55 * np.sin(angles2)],
        np.c_[0.25 * np.cos(angles3), 0.25 * np.sin(angles3)]
    ])
    inits.append(ring3)
    
    # 5. Random inits
    np.random.seed(42)
    for _ in range(8):
        inits.append(np.random.randn(n, d))
    
    # Normalize all inits
    inits_norm = [_normalize_points(init) for init in inits]
    
    # Multi-start Powell optimization with smooth surrogate
    for idx, x0 in enumerate(inits_norm):
        if time.time() > deadline:
            break
        
        x0_flat = x0.flatten()
        
        # Powell optimization (derivative-free, good for non-smooth)
        try:
            res_powell = opt.minimize(
                _soft_log_min_max, x0_flat, method='Powell',
                options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-10, 'maxfev': 2000}
            )
            points_powell = res_powell.x.reshape(n, d)
            score_p = update_best(points_powell)
        except Exception:
            pass
        
        # L-BFGS-B with analytic gradient
        if time.time() > deadline:
            break
        try:
            res_lbfgs = opt.minimize(
                _soft_log_min_max, x0_flat, method='L-BFGS-B',
                jac=_soft_log_min_max_grad,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            points_lbfgs = res_lbfgs.x.reshape(n, d)
            score_l = update_best(points_lbfgs)
        except Exception:
            pass
        
        # Nelder-Mead polish on the non-smooth objective
        if time.time() > deadline:
            break
        # Polish from the best of Powell and L-BFGS-B results
        candidates = [x0]
        if 'res_powell' in dir() and res_powell.x is not None:
            candidates.append(res_powell.x.reshape(n, d))
        if 'res_lbfgs' in dir() and res_lbfgs.x is not None:
            candidates.append(res_lbfgs.x.reshape(n, d))
        
        # Pick the best candidate to polish
        best_cand = candidates[0]
        best_cand_score = -np.inf
        for cand in candidates:
            s = calculate_score(cand)
            if s > best_cand_score:
                best_cand_score = s
                best_cand = cand
        
        points_polished = _nelder_mead_polish(best_cand, maxiter=100, tol=1e-9)
        update_best(points_polished)
    
    # Additional random restarts if time permits
    np.random.seed(123)
    for _ in range(5):
        if time.time() > deadline:
            break
        x0_rand = _normalize_points(np.random.randn(n, d))
        try:
            res = opt.minimize(
                _soft_log_min_max, x0_rand.flatten(), method='Powell',
                options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-10, 'maxfev': 1500}
            )
            points_r = res.x.reshape(n, d)
            update_best(points_r)
        except Exception:
            pass
    
    # Final Nelder-Mead polish on the best found
    if best_points is not None and time.time() < deadline - 10:
        best_polished = _nelder_mead_polish(best_points, maxiter=150, tol=1e-10)
        update_best(best_polished)
    
    # Fallback
    if best_points is None:
        for init in inits:
            update_best(init)
    
    return best_points

# EVOLVE-BLOCK-END
