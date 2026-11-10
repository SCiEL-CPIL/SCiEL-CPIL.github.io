# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _fibonacci_sphere(n):
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def _objective_and_grad(x, beta):
    """
    Objective to minimize: -log(d_min) + log(d_max)
    Approximated by:
    logsumexp(-dist^2 / beta) - logsumexp(dist^2 / beta)
    
    Minimizing this maximizes d_min/d_max.
    Note: We use squared distances to make it smoother and scale-invariant in a way?
    Actually, let's stick to distances for clarity, but squared is often numerically better.
    Let's use squared distances D = dist^2.
    Maximize D_min / D_max.
    Minimize log(D_max) - log(D_min).
    
    Approximation:
    log(D_max) ~ logsumexp(D / beta) - log(beta)
    log(D_min) ~ -logsumexp(-D / beta) + log(beta)
    
    So Minimize: logsumexp(D / beta) - logsumexp(-D / beta)
    """
    n = 14
    pts = x.reshape(n, 3)
    
    # Compute all pairwise squared distances
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Zero out diagonal
    np.fill_diagonal(dist_sq, 0)
    
    # We only need upper triangle or just all, logsumexp handles it.
    # Using all is fine, just double the count.
    D = dist_sq
    
    # For stability in logsumexp, subtract max
    # logsumexp(D / beta)
    D_pos = D / beta
    max_pos = np.max(D_pos)
    # If max is 0, all are 0, logsumexp is log(n)
    if max_pos == 0:
        lse_pos = np.log(np.sum(np.exp(D_pos)))
    else:
        lse_pos = max_pos + np.log(np.sum(np.exp(D_pos - max_pos)))
        
    # logsumexp(-D / beta)
    D_neg = -D / beta
    max_neg = np.max(D_neg)
    if max_neg == 0:
        lse_neg = np.log(np.sum(np.exp(D_neg)))
    else:
        lse_neg = max_neg + np.log(np.sum(np.exp(D_neg - max_neg)))
        
    val = lse_pos - lse_neg
    
    # Gradient
    # d/dx logsumexp(f(x)) = sum(exp(f_i - max_f) * f_i') / sum(exp(f_i - max_f))
    
    # For lse_pos: f_i = D_i / beta
    # D_i = ||p_i - p_j||^2
    # d D_i / d p_k = 2(p_k - p_j) if k=i, 2(p_k - p_i) if k=j, 0 otherwise
    
    # We need to compute gradient efficiently.
    # Let's compute weights for pos and neg parts.
    
    # Weights for pos part (max D)
    if max_pos == 0:
        w_pos = np.ones_like(D) / (n*(n-1)) # uniform if all 0? No, if all 0, grad is 0?
        # If all dists are 0, points coincide. Gradient of logsumexp(0) is uniform.
        # But D_i=0 for all i.
        # Actually if points coincide, D is all 0.
        # exp(0)=1. sum = N.
        # w_i = 1/N.
        # Grad D_i wrt p_k.
    else:
        exp_pos = np.exp(D_pos - max_pos)
        sum_exp_pos = np.sum(exp_pos)
        w_pos = exp_pos / sum_exp_pos
        
    # Weights for neg part (min D)
    # f_i = -D_i / beta
    if max_neg == 0:
        w_neg = np.ones_like(D) / (n*(n-1))
    else:
        exp_neg = np.exp(D_neg - max_neg)
        sum_exp_neg = np.sum(exp_neg)
        w_neg = exp_neg / sum_exp_neg
        
    # Objective gradient: grad(lse_pos) - grad(lse_neg)
    # grad(lse_pos) = (1/beta) * sum_i w_pos[i] * grad(D_i)
    # grad(lse_neg) = (1/beta) * sum_i w_neg[i] * grad(-D_i) = -(1/beta) * sum_i w_neg[i] * grad(D_i)
    
    # So total grad = (1/beta) * [ sum_i w_pos[i] grad(D_i) + sum_i w_neg[i] grad(D_i) ]
    # Let W = w_pos + w_neg
    # Grad = (1/beta) * sum_i W[i] grad(D_i)
    
    W = w_pos + w_neg
    
    # grad(D_i) wrt p_k
    # D_i is dist_sq between pair (u, v).
    # We need to sum over all pairs (u, v) weighted by W[u, v].
    # W is symmetric.
    # For each point k, grad_k = sum_{u,v} W[u,v] * dD_uv/dp_k
    
    # dD_uv/dp_k:
    # if k == u: 2(p_u - p_v)
    # if k == v: 2(p_v - p_u) = -2(p_u - p_v)
    # else 0
    
    # So grad_k = 2 * [ sum_{v != k} W[k, v] (p_k - p_v) - sum_{u != k} W[u, k] (p_u - p_k) ]
    # Since W is symmetric, W[k, v] = W[v, k].
    # Sum over v of W[k, v] (p_k - p_v) = p_k * sum_v W[k, v] - sum_v W[k, v] p_v
    # Let row_sum_k = sum_v W[k, v]
    # Let w_dot_p_k = sum_v W[k, v] p_v
    
    # grad_k = 2 * [ p_k * row_sum_k - w_dot_p_k - (w_dot_p_k - p_k * row_sum_k) ]
    # Wait.
    # Term 1: sum_{v} W[k, v] (p_k - p_v) = p_k S_k - P_k
    # Term 2: - sum_{u} W[u, k] (p_u - p_k) = - (P_k - p_k S_k) = p_k S_k - P_k
    # So grad_k = 2 * [ 2(p_k S_k - P_k) ] = 4 (p_k S_k - P_k)
    
    S = np.sum(W, axis=1) # shape (n,)
    P = W @ pts # shape (n, 3)
    
    grad_pts = 4.0 * (pts * S[:, np.newaxis] - P)
    grad = (1.0 / beta) * grad_pts.flatten()
    
    return val, grad

def _two_opt(points):
    """
    Apply 2-opt moves to improve the score.
    Since the objective is symmetric under permutation, we can swap any two points.
    We try all swaps and keep the best improvement.
    """
    n = 14
    pts = points.copy()
    best_score = _score(pts)
    
    improved = True
    while improved:
        improved = False
        best_swap_score = best_score
        best_swap = None
        
        for i in range(n):
            for j in range(i+1, n):
                new_pts = pts.copy()
                new_pts[i], new_pts[j] = new_pts[j], new_pts[i]
                s = _score(new_pts)
                if s > best_swap_score + 1e-12:
                    best_swap_score = s
                    best_swap = (i, j)
        
        if best_swap:
            i, j = best_swap
            pts[i], pts[j] = pts[j], pts[i]
            best_score = best_swap_score
            improved = True
            
    return pts, best_score

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    n = 14
    
    # Initial guess: Fibonacci sphere
    pts = _fibonacci_sphere(n)
    
    # Normalize to unit sphere initially
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # We will try multiple beta values and restarts
    # Beta controls the sharpness of the approximation.
    # Small beta -> closer to true min/max.
    # Large beta -> smoother, easier to optimize but less accurate.
    
    betas = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        # Start from current best
        x0 = best_pts.flatten()
        
        # Run L-BFGS-B
        # We minimize the objective
        # The objective is smooth, so L-BFGS-B should work well.
        
        res = minimize(
            _objective_and_grad, 
            x0, 
            args=(beta,), 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if res.success or res.status == 0:
            new_pts = res.x.reshape(n, 3)
            # Re-center and normalize scale to avoid drift
            new_pts -= np.mean(new_pts, axis=0)
            scale = np.linalg.norm(new_pts, axis=1)
            # If scale is 0, skip
            if np.any(scale == 0):
                continue
            # Optional: normalize to unit sphere?
            # The problem is scale invariant.
            # However, keeping points on a sphere might help with the next iteration?
            # No, the optimum might not be on a sphere.
            # Just keep as is, but ensure finite.
            if not np.all(np.isfinite(new_pts)):
                continue
                
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                
        # After L-BFGS, try 2-opt to fix any permutation issues or local minima
        # 2-opt is expensive (91 swaps * score eval). Score eval is fast.
        # Let's do a few rounds of 2-opt.
        
        if time.time() < deadline - 5:
            polished_pts, polished_score = _two_opt(best_pts)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
                
    # Final polish with a very small beta if time remains
    if time.time() < deadline - 10:
        beta = 0.1
        x0 = best_pts.flatten()
        res = minimize(
            _objective_and_grad, 
            x0, 
            args=(beta,), 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.success or res.status == 0:
            new_pts = res.x.reshape(n, 3)
            new_pts -= np.mean(new_pts, axis=0)
            if np.all(np.isfinite(new_pts)):
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    
    return best_pts

# EVOLVE-BLOCK-END
