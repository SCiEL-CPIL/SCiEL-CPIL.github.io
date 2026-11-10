"""Packing 50 equal circles in a unit circle: ring init + gradient polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
SEED_R = 0.1258254895304  # best known packing radius (for scoring baseline)


def _score(points):
    """Compute the common radius r for a set of centres."""
    n = len(points)
    # container constraint
    r_cont = 1.0 - np.linalg.norm(points, axis=1).min()
    # pairwise distances
    d = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0
    return min(r_cont, r_pair)


def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    n = len(points)
    # container
    r_cont = 1.0 - np.linalg.norm(points, axis=1).min()
    # pairwise
    d = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0
    return -min(r_cont, r_pair)


def _neg_score_grad(points_flat):
    """Negative score with gradient."""
    points = points_flat.reshape(-1, 2)
    n = len(points)
    # container
    norms = np.linalg.norm(points, axis=1)
    r_cont = 1.0 - norms.min()
    # pairwise
    d = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0

    r = min(r_cont, r_pair)
    if r <= 0:
        # degenerate; push apart
        grad = np.zeros_like(points_flat)
        return -r, grad

    # Gradient: we want to maximize min(r_cont, r_pair)
    # If r_cont < r_pair, gradient w.r.t. points is from container constraint
    # If r_pair < r_cont, gradient from the closest pair
    if r_cont < r_pair:
        # r_cont = 1 - min_i |p_i|
        # d r_cont / d p_i = - p_i / |p_i|  for the closest to boundary
        # We want to MAXIMIZE r, so gradient of -r = + p_i/|p_i| for closest
        idx = np.argmin(norms)
        grad = np.zeros_like(points_flat)
        p = points[idx]
        norm = norms[idx]
        if norm > 1e-15:
            grad[2*idx:2*idx+2] = p / norm
        return -r, grad
    else:
        # r_pair = min_{i<j} |p_i - p_j| / 2
        # Find closest pair
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        diff = points[i] - points[j]
        dist_ij = np.linalg.norm(diff)
        if dist_ij < 1e-15:
            grad = np.zeros_like(points_flat)
            return -r, grad
        # d r_pair / d p_i = -diff / (2 * dist_ij) ... wait
        # r_pair = dist_ij / 2
        # d r_pair / d p_i = (p_i - p_j) / (2 * dist_ij)
        # d r_pair / d p_j = (p_j - p_i) / (2 * dist_ij)
        # We maximize r, so gradient of -r = -gradient of r
        # grad of -r w.r.t p_i = -(p_i - p_j) / (2 * dist_ij)
        # grad of -r w.r.t p_j = -(p_j - p_i) / (2 * dist_ij) = (p_i - p_j) / (2 * dist_ij)
        grad = np.zeros_like(points_flat)
        unit = diff / dist_ij
        grad[2*i:2*i+2] = -unit / 2.0
        grad[2*j:2*j+2] = unit / 2.0
        return -r, grad


def _ring_init(n):
    """
    Initialize N circles in concentric rings matching known structure:
    1 + 5 + 13 + 31 = 50
    Ring 0: 1 circle at center
    Ring 1: 5 circles
    Ring 2: 13 circles
    Ring 3: 31 circles
    """
    points = []
    # Ring 0: center
    points.append([0.0, 0.0])
    
    # Ring 1: 5 circles
    n1 = 5
    r1 = 0.25
    for i in range(n1):
        angle = 2 * np.pi * i / n1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 13 circles
    n2 = 13
    r2 = 0.5
    for i in range(n2):
        angle = 2 * np.pi * i / n2 + np.pi / n2  # offset to avoid alignment
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 31 circles
    n3 = 31
    r3 = 0.75
    for i in range(n3):
        angle = 2 * np.pi * i / n3
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    return np.array(points[:n])


def _random_init(n, seed=42):
    """Random initialization inside unit circle."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(n):
        while True:
            p = rng.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 0.9:
                points.append(p)
                break
    return np.array(points[:n])


def _optimize(points, max_time=100.0, tol=1e-12):
    """
    Optimize the packing using gradient-based method.
    Uses a smooth approximation: maximize the minimum of all constraints.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    x = points.flatten()
    
    # Use a soft-min approach: instead of hard min, use a smooth approximation
    # f(x) = -logsumexp(-alpha * constraints) / alpha  approximates min(constraints)
    # But simpler: just use the hard min with gradient as above, with L-BFGS-B
    
    # Actually, let's use a different approach: 
    # We want to maximize r subject to:
    #   |p_i - p_j| >= 2r for all i<j
    #   |p_i| + r <= 1 for all i
    # 
    # This is equivalent to: given r, check feasibility. But we want to find max r.
    # 
    # Alternative: use the fact that the objective is the min of linear-ish functions.
    # Let's just do gradient ascent on the smooth surrogate.
    
    # Smooth surrogate: use log-sum-exp to approximate the min
    # g(x) = -(1/alpha) * log(sum_i exp(-alpha * c_i(x)))
    # where c_i are the constraint values (distances/2 and 1-|p_i|)
    # As alpha -> inf, g -> min(c_i)
    
    def soft_min_obj(x_flat, alpha=50.0):
        points = x_flat.reshape(-1, 2)
        n = len(points)
        
        # Container constraints: c_i = 1 - |p_i|
        norms = np.linalg.norm(points, axis=1)
        c_cont = 1.0 - norms
        
        # Pairwise constraints: c_ij = |p_i - p_j| / 2
        d = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(d, axis=2)
        np.fill_diagonal(dist, np.inf)
        c_pair = dist / 2.0
        
        # Combine all constraints
        # Use log-sum-exp for numerical stability
        all_c = np.concatenate([c_cont, c_pair[c_pair < np.inf].flatten()])
        
        # Soft min: -(1/alpha) * log(sum(exp(-alpha * c)))
        # For numerical stability:
        max_c = all_c.max()
        shifted = -alpha * (all_c - max_c)
        # Clip for overflow
        shifted = np.clip(shifted, -700, 700)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min = max_c - log_sum / alpha
        
        return -soft_min  # We minimize, so negate
    
    def soft_min_grad(x_flat, alpha=50.0):
        points = x_flat.reshape(-1, 2)
        n = len(points)
        
        # Container constraints
        norms = np.linalg.norm(points, axis=1)
        c_cont = 1.0 - norms
        
        # Pairwise constraints
        d = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(d, axis=2)
        np.fill_diagonal(dist, np.inf)
        c_pair = dist / 2.0
        pair_mask = c_pair < np.inf
        c_pair_flat = c_pair[pair_mask]
        
        all_c = np.concatenate([c_cont, c_pair_flat])
        
        # Soft min gradient
        max_c = all_c.max()
        shifted = -alpha * (all_c - max_c)
        shifted = np.clip(shifted, -700, 700)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp  # softmax weights
        
        # Gradient of soft_min w.r.t. x
        # soft_min = max_c - (1/alpha) * log(sum(exp(-alpha * c_i)))
        # d soft_min / d x = - (1/alpha) * sum_i weights_i * (-alpha * d c_i / d x)
        #                   = sum_i weights_i * d c_i / d x
        
        grad = np.zeros_like(x_flat)
        
        # Container part
        for i in range(n):
            w = weights[i]
            p = points[i]
            norm = norms[i]
            if norm > 1e-15:
                # d c_i / d p_i = -p_i / |p_i|
                grad[2*i:2*i+2] += w * (-p / norm)
        
        # Pairwise part
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                w = weights[n + idx]
                diff = points[i] - points[j]
                dist_ij = np.linalg.norm(diff)
                if dist_ij > 1e-15:
                    # d c_ij / d p_i = (p_i - p_j) / (2 * dist_ij)
                    # d c_ij / d p_j = (p_j - p_i) / (2 * dist_ij)
                    unit = diff / dist_ij
                    grad[2*i:2*i+2] += w * unit / 2.0
                    grad[2*j:2*j+2] += w * (-unit / 2.0)
                idx += 1
        
        # We minimize -soft_min, so gradient is -grad
        return -grad
    
    # Start with L-BFGS-B
    best_x = x.copy()
    best_val = soft_min_obj(x)
    
    # Try multiple alpha values
    for alpha in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 5:
            break
        
        res = minimize(
            soft_min_obj, x, 
            jac=lambda xf: soft_min_grad(xf, alpha),
            args=(alpha,),
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000}
        )
        
        val = soft_min_obj(res.x)
        if val < best_val:
            best_val = val
            best_x = res.x.copy()
        
        x = res.x  # Continue from best
    
    # Final polish with hard min and smaller steps
    if time.time() < deadline - 2:
        # Do a few more iterations with higher alpha
        for alpha in [500.0, 1000.0]:
            if time.time() > deadline - 1:
                break
            res = minimize(
                soft_min_obj, best_x,
                jac=lambda xf: soft_min_grad(xf, alpha),
                args=(alpha,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
            )
            val = soft_min_obj(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
    
    return best_x.reshape(-1, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try ring-based initialization first (matches known optimal structure)
    points_ring = _ring_init(N)
    
    best_points = points_ring
    best_score = _score(points_ring)
    
    # Optimize from ring init
    if time.time() < deadline - 10:
        opt_points = _optimize(points_ring, max_time=60.0)
        s = _score(opt_points)
        if s > best_score:
            best_score = s
            best_points = opt_points
    
    # Try a few random inits if time permits
    if time.time() < deadline - 30:
        for seed in [1, 7, 13, 42, 123]:
            if time.time() > deadline - 10:
                break
            points_rand = _random_init(N, seed=seed)
            s_rand = _score(points_rand)
            if s_rand > best_score:
                best_score = s_rand
                best_points = points_rand
            
            # Optimize from random init
            if time.time() < deadline - 15:
                opt_rand = _optimize(points_rand, max_time=min(30.0, deadline - time.time() - 5))
                s_opt = _score(opt_rand)
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = opt_rand
    
    # Final check: ensure validity
    n = len(best_points)
    if n != N:
        # Fallback to ring init
        best_points = _ring_init(N)
    
    # Verify all points are finite and inside
    if not np.all(np.isfinite(best_points)):
        best_points = _ring_init(N)
    
    # Ensure points are strictly inside (with small margin)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms < 0.9999
    if not np.all(mask):
        # Scale down slightly
        max_norm = norms.max()
        if max_norm > 0:
            best_points *= 0.9999 / max_norm
    
    return best_points
