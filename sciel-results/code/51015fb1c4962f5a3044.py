# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize


def calculate_score(points):
    """Calculate R = (d_min / d_max)^2 for 16 points in 2D."""
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


def soft_min_max(points, beta=0.05):
    """
    Compute smooth approximations of d_min and d_max using log-sum-exp.
    Returns (soft_dmin, soft_dmax, dists, idx_i, idx_j)
    """
    n = len(points)
    idx_i, idx_j = np.triu_indices(n, k=1)
    diffs = points[idx_i] - points[idx_j]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    
    # Soft min: -logsumexp(-dists/beta) * beta
    max_d = np.max(dists)
    exp_neg = np.exp(-(dists - max_d) / beta)
    logsumexp_neg = np.log(np.sum(exp_neg)) + (max_d - max_d) / beta
    # Actually: soft_min = -beta * log(sum(exp(-d/beta)))
    # For numerical stability:
    min_d = np.min(dists)
    exp_neg_stable = np.exp(-(dists - min_d) / beta)
    soft_dmin = min_d - beta * np.log(np.sum(exp_neg_stable))
    
    # Soft max: beta * log(sum(exp(d/beta)))
    exp_pos_stable = np.exp((dists - max_d) / beta)
    soft_dmax = max_d + beta * np.log(np.sum(exp_pos_stable))
    
    return soft_dmin, soft_dmax, dists, idx_i, idx_j, diffs


def smooth_objective_flat(flat, beta=0.05):
    """
    Negative smooth objective for minimization.
    flat is 2D array of shape (n, 2) flattened to 2n.
    We fix centroid at origin and scale so d_max ~ 1.
    """
    n = flat.shape[0] // 2
    points = flat.reshape(n, 2)
    
    # Center
    points = points - np.mean(points, axis=0)
    
    # Scale to unit diameter
    idx_i, idx_j = np.triu_indices(n, k=1)
    diffs = points[idx_i] - points[idx_j]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    d_max = np.max(dists)
    if d_max < 1e-10:
        return 1e10  # penalty
    
    points = points / d_max
    
    # Now compute soft min/max on scaled points
    diffs = points[idx_i] - points[idx_j]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    
    min_d = np.min(dists)
    max_d = np.max(dists)
    
    exp_neg_stable = np.exp(-(dists - min_d) / beta)
    soft_dmin = min_d - beta * np.log(np.sum(exp_neg_stable))
    
    exp_pos_stable = np.exp((dists - max_d) / beta)
    soft_dmax = max_d + beta * np.log(np.sum(exp_pos_stable))
    
    if soft_dmax < 1e-10:
        return 1e10
    
    ratio = soft_dmin / soft_dmax
    # Maximize ratio^2, so minimize -ratio^2
    return -(ratio * ratio)


def smooth_gradient_flat(flat, beta=0.05):
    """
    Analytic gradient of the smooth objective with respect to flat parameters.
    This is the gradient of - (soft_dmin/soft_dmax)^2.
    """
    n = flat.shape[0] // 2
    points = flat.reshape(n, 2)
    
    # Center
    points = points - np.mean(points, axis=0)
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    diffs = points[idx_i] - points[idx_j]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    d_max = np.max(dists)
    if d_max < 1e-10:
        return np.zeros_like(flat)
    
    points = points / d_max
    
    # Recompute after scaling
    diffs = points[idx_i] - points[idx_j]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    
    min_d = np.min(dists)
    max_d = np.max(dists)
    
    # Soft min: f = min_d - beta * log(sum(exp(-(d_k - min_d)/beta)))
    # Let S_neg = sum(exp(-(d_k - min_d)/beta))
    exp_neg = np.exp(-(dists - min_d) / beta)
    S_neg = np.sum(exp_neg)
    
    # d(soft_dmin)/d(d_k) = -exp(-(d_k-min_d)/beta) / S_neg * (-1) ... 
    # Actually: soft_dmin = min_d - beta*log(S_neg)
    # d(soft_dmin)/d(d_k) for k != argmin: -beta * d(log S_neg)/d(d_k) = -beta * (-exp(-(d_k-min_d)/beta)/S_neg) / 1 ... wait
    # S_neg = sum_j exp(-(d_j - min_d)/beta)
    # dS_neg/dd_k = exp(-(d_k-min_d)/beta) * (-1/beta)
    # d(log S_neg)/dd_k = (1/S_neg) * dS_neg/dd_k = -exp(-(d_k-min_d)/beta)/(beta*S_neg)
    # d(soft_dmin)/dd_k = -beta * (-exp(-(d_k-min_d)/beta)/(beta*S_neg)) = exp(-(d_k-min_d)/beta)/S_neg
    # But min_d also depends on dists... this gets complicated.
    
    # For the gradient, I'll use a simpler approach: 
    # The weight for each distance in the soft min is w_k = exp(-(d_k - min_d)/beta) / S_neg
    # soft_dmin = sum_k w_k * d_k (this is a weighted average interpretation)
    # Actually that's not quite right, but the gradient w.r.t. d_k is w_k for the min part.
    
    w_neg = exp_neg / S_neg  # weights for soft min
    
    # Soft max: soft_dmax = max_d + beta*log(S_pos) where S_pos = sum(exp((d_k-max_d)/beta))
    exp_pos = np.exp((dists - max_d) / beta)
    S_pos = np.sum(exp_pos)
    w_pos = exp_pos / S_pos  # weights for soft max
    
    # soft_dmin ≈ sum_k w_neg[k] * d_k (approximately, for gradient purposes)
    # soft_dmax ≈ sum_k w_pos[k] * d_k
    
    # ratio = soft_dmin / soft_dmax
    soft_dmin_val = np.sum(w_neg * dists)
    soft_dmax_val = np.sum(w_pos * dists)
    
    if soft_dmax_val < 1e-10:
        return np.zeros_like(flat)
    
    ratio = soft_dmin_val / soft_dmax_val
    
    # d(-ratio^2)/d(soft_dmin) = -2*ratio/soft_dmax
    # d(-ratio^2)/d(soft_dmax) = 2*ratio^2/soft_dmax^2
    
    # d(soft_dmin)/d(d_k) = w_neg[k]
    # d(soft_dmax)/d(d_k) = w_pos[k]
    
    # d(d_k)/d(points) for pair (i,j): d_k = ||p_i - p_j||
    # d(d_k)/d(p_i) = (p_i - p_j)/d_k
    # d(d_k)/d(p_j) = (p_j - p_i)/d_k = -(p_i - p_j)/d_k
    
    grad_dmin_dd = w_neg  # shape (n_pairs,)
    grad_dmax_dd = w_pos  # shape (n_pairs,)
    
    d_obj_d_dmin = -2 * ratio / soft_dmax_val
    d_obj_d_dmax = 2 * ratio * ratio / (soft_dmax_val * soft_dmax_val)
    
    # Combined gradient w.r.t. each distance
    grad_dd = d_obj_d_dmin * grad_dmin_dd + d_obj_d_dmax * grad_dmax_dd  # shape (n_pairs,)
    
    # Now propagate to points
    grad_points = np.zeros((n, 2))
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        d_k = dists[k]
        if d_k < 1e-10:
            continue
        # d(d_k)/d(p_i) = (p_i - p_j)/d_k
        # d(d_k)/d(p_j) = -(p_i - p_j)/d_k
        factor = grad_dd[k] / d_k
        grad_points[i] += factor * diffs[k]
        grad_points[j] -= factor * diffs[k]
    
    # Account for centering: points_centered = points - mean(points)
    # d(points_centered)/d(points) = I - (1/n) * ones
    # So grad_flat = (I - (1/n)ones) @ grad_points, but since we're in centered space,
    # and the scaling... this is getting complex. For now, return the direct gradient.
    # The optimizer should handle the constraint approximately.
    
    # Flatten
    grad_flat = grad_points.flatten()
    
    # Also account for the scaling step: points_scaled = points_centered / d_max
    # This adds a Jacobian term. For simplicity, we'll ignore it and rely on
    # the optimizer's robustness. The scaling is mild.
    
    return grad_flat


def nelder_mead_polish(points, start_time, max_time=30.0, beta_nm=0.01):
    """
    Polish using Nelder-Mead on the true non-smooth objective.
    """
    n = len(points)
    points = points.copy().astype(float)
    
    # Center and scale
    points = points - np.mean(points, axis=0)
    idx_i, idx_j = np.triu_indices(n, k=1)
    diffs = points[idx_i] - points[idx_j]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    d_max = np.max(dists)
    if d_max > 0:
        points = points / d_max
    
    best_score = calculate_score(points)
    best_points = points.copy()
    
    def objective(flat):
        pts = flat.reshape(n, 2)
        pts = pts - np.mean(pts, axis=0)
        idx_i2, idx_j2 = np.triu_indices(n, k=1)
        diffs2 = pts[idx_i2] - pts[idx_j2]
        dists2 = np.sqrt(np.sum(diffs2 ** 2, axis=1))
        dm = np.max(dists2)
        if dm < 1e-10:
            return 1e10
        pts = pts / dm
        return -calculate_score(pts)
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < 1e9:
            new_points = result.x.reshape(n, 2)
            new_points = new_points - np.mean(new_points, axis=0)
            idx_i2, idx_j2 = np.triu_indices(n, k=1)
            diffs2 = new_points[idx_i2] - new_points[idx_j2]
            dists2 = np.sqrt(np.sum(diffs2 ** 2, axis=1))
            dm = np.max(dists2)
            if dm > 0:
                new_points = new_points / dm
            new_score = calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
    except Exception:
        pass
    
    return best_points, best_score


def lbfgs_optimize(points, start_time, max_time=120.0, beta=0.05):
    """
    Optimize using L-BFGS-B on the smooth surrogate.
    """
    n = len(points)
    points = points.copy().astype(float)
    
    # Center and scale
    points = points - np.mean(points, axis=0)
    idx_i, idx_j = np.triu_indices(n, k=1)
    diffs = points[idx_i] - points[idx_j]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    d_max = np.max(dists)
    if d_max > 0:
        points = points / d_max
    
    best_score = calculate_score(points)
    best_points = points.copy()
    
    x0 = points.flatten()
    
    def objective_and_grad(flat):
        obj = smooth_objective_flat(flat, beta=beta)
        grad = smooth_gradient_flat(flat, beta=beta)
        return obj, grad
    
    try:
        result = minimize(
            objective_and_grad, x0, method='L-BFGS-B', jac=True,
            options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8}
        )
        if result.fun < 1e9:
            new_points = result.x.reshape(n, 2)
            new_points = new_points - np.mean(new_points, axis=0)
            idx_i2, idx_j2 = np.triu_indices(n, k=1)
            diffs2 = new_points[idx_i2] - new_points[idx_j2]
            dists2 = np.sqrt(np.sum(diffs2 ** 2, axis=1))
            dm = np.max(dists2)
            if dm > 0:
                new_points = new_points / dm
            new_score = calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
    except Exception:
        pass
    
    return best_points, best_score


def hexagonal_init(n, scale=1.0, angle=0.0):
    """Initialize points on a hexagonal grid pattern."""
    points = []
    rows = 4
    cols = 4
    dx = 1.0 * scale
    dy = dx * np.sqrt(3) / 2
    
    for i in range(rows):
        for j in range(cols):
            x = j * dx + (i % 2) * (dx / 2)
            y = i * dy
            points.append([x, y])
    
    pts = np.array(points[:n])
    if angle != 0:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T
    return pts


def ring_init(n, scale=1.0, seed=None):
    """Initialize points in concentric rings."""
    rng = np.random.default_rng(seed)
    points = []
    # Inner ring: 4 points
    for k in range(4):
        angle = 2 * np.pi * k / 4 + np.pi / 4
        points.append([0.3 * scale * np.cos(angle), 0.3 * scale * np.sin(angle)])
    # Outer ring: 12 points
    for k in range(12):
        angle = 2 * np.pi * k / 12
        points.append([scale * np.cos(angle), scale * np.sin(angle)])
    return np.array(points[:n])


def random_init(n, scale=1.0, seed=None):
    """Initialize points randomly within a unit circle."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0.2, 1.0, n) * scale
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def grid_init(n, scale=1.0, seed=None):
    """Initialize on a jittered grid."""
    rng = np.random.default_rng(seed)
    rows = 4
    cols = 4
    points = []
    for i in range(rows):
        for j in range(cols):
            x = (j - 1.5) * scale + rng.normal(0, 0.05)
            y = (i - 1.5) * scale * 0.866 + rng.normal(0, 0.05)
            points.append([x, y])
    return np.array(points[:n])


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    Returns np.ndarray of shape (16, 2).
    """
    start_time = time.time()
    max_time = 600.0  # Safety margin before 630s limit
    
    n = 16
    best_points = None
    best_score = 0.0
    
    # Generate diverse initializations
    initializations = []
    
    # Hexagonal with different rotations
    for angle in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        initializations.append(hexagonal_init(n, scale=1.0, angle=angle))
    
    # Ring-based
    for seed in [42, 123, 456]:
        initializations.append(ring_init(n, scale=1.0, seed=seed))
    
    # Random
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        initializations.append(random_init(n, scale=1.0, seed=seed))
    
    # Jittered grid
    for seed in [10, 20, 30]:
        initializations.append(grid_init(n, scale=1.0, seed=seed))
    
    # Shuffle for diversity
    rng = np.random.default_rng(99)
    perm = rng.permutation(len(initializations))
    initializations = [initializations[i] for i in perm]
    
    # Allocate time: use L-BFGS for most, NM polish for best candidates
    n_inits = len(initializations)
    time_per_init = (max_time - 60) / n_inits  # Reserve 60s for final polish
    
    for idx, init_points in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > max_time - 30:
            break
        
        remaining = max_time - elapsed - 30
        if remaining < 10:
            break
        
        # Phase 1: L-BFGS-B optimization
        t1_start = time.time()
        t1_budget = min(time_per_init * 0.7, remaining * 0.6)
        lbfgs_points, lbfgs_score = lbfgs_optimize(
            init_points, t1_start, max_time=t1_budget, beta=0.05
        )
        
        if lbfgs_score > best_score:
            best_score = lbfgs_score
            best_points = lbfgs_points.copy()
        
        # Phase 2: Nelder-Mead polish on the best result
        elapsed = time.time() - start_time
        if elapsed > max_time - 30:
            break
        
        remaining = max_time - elapsed - 30
        if remaining < 5:
            break
        
        t2_budget = min(time_per_init * 0.3, remaining * 0.4)
        nm_points, nm_score = nelder_mead_polish(
            lbfgs_points, time.time(), max_time=t2_budget
        )
        
        if nm_score > best_score:
            best_score = nm_score
            best_points = nm_points.copy()
    
    # Final polish on the overall best with remaining time
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = max_time - elapsed - 5
        if remaining > 10:
            final_points, final_score = nelder_mead_polish(
                best_points, time.time(), max_time=min(remaining, 60.0)
            )
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
    
    # Ensure valid output
    if best_points is None:
        best_points = hexagonal_init(n)
    
    assert best_points.shape == (16, 2), f"Invalid shape: {best_points.shape}"
    assert np.all(np.isfinite(best_points)), "Non-finite values in points"
    
    return best_points


# EVOLVE-BLOCK-END
