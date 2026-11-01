"""Improved program using SDP/Gram matrix approach for n=14, d=3."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    # Compute all pairwise squared distances
    d_sq = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.dot(points[i] - points[j], points[i] - points[j])
            if d > 1e-10:
                d_sq.append(d)
    if not d_sq:
        return 1e10
    min_d_sq = min(d_sq)
    max_d_sq = max(d_sq)
    if min_d_sq < 1e-10:
        return 1e10
    return (max_d_sq / min_d_sq)

def _gram_to_points(G, d):
    """Extract points from Gram matrix via eigenvalue decomposition."""
    eigvals, eigvecs = np.linalg.eigh(G)
    # Take top d eigenvalues/eigenvectors
    idx = np.argsort(eigvals)[::-1][:d]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Points are eigvecs * sqrt(eigvals)
    points = eigvecs * np.sqrt(np.maximum(eigvals, 0))
    # Center the points
    points = points - points.mean(axis=0)
    return points

def _gram_objective(x, n, d, lambda_psd, lambda_rank):
    """
    Objective function for Gram matrix optimization.
    x: flattened Gram matrix (n*n,)
    lambda_psd: penalty for negative eigenvalues
    lambda_rank: penalty for eigenvalues beyond top d
    """
    G = x.reshape(n, n)
    # Ensure symmetry
    G = (G + G.T) / 2.0
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(G)
    
    # PSD penalty: sum of squares of negative eigenvalues
    neg_eigs = eigvals[eigvals < 0]
    psd_penalty = lambda_psd * np.sum(neg_eigs ** 2)
    
    # Rank penalty: sum of squares of eigenvalues beyond top d
    sorted_eigs = np.sort(eigvals)[::-1]
    if d < len(sorted_eigs):
        rank_penalty = lambda_rank * np.sum(sorted_eigs[d:] ** 2)
    else:
        rank_penalty = 0.0
    
    # Distance matrix from Gram matrix: D_ij = G_ii + G_jj - 2*G_ij
    d_sq = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = G[i, i] + G[j, j] - 2 * G[i, j]
            d_sq[i, j] = val
            d_sq[j, i] = val
    
    # Extract off-diagonal squared distances
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            if d_sq[i, j] > 1e-12:
                off_diag.append(d_sq[i, j])
    
    if not off_diag:
        return 1e10 + psd_penalty + rank_penalty
    
    min_d_sq = min(off_diag)
    max_d_sq = max(off_diag)
    
    if min_d_sq < 1e-12:
        return 1e10 + psd_penalty + rank_penalty
    
    # Ratio of squared distances
    ratio = max_d_sq / min_d_sq
    
    return ratio + psd_penalty + rank_penalty

def _gram_grad(x, n, d, lambda_psd, lambda_rank):
    """
    Gradient of the Gram matrix objective.
    """
    G = x.reshape(n, n)
    G = (G + G.T) / 2.0
    
    eigvals, eigvecs = np.linalg.eigh(G)
    
    # PSD gradient
    neg_mask = eigvals < 0
    psd_grad = 2 * lambda_psd * (eigvecs * neg_mask) @ eigvecs.T
    
    # Rank gradient
    sorted_idx = np.argsort(eigvals)[::-1]
    rank_grad = np.zeros((n, n))
    if d < n:
        for i in range(d, n):
            rank_grad += 2 * lambda_rank * eigvals[sorted_idx[i]] * np.outer(eigvecs[:, sorted_idx[i]], eigvecs[:, sorted_idx[i]])
    
    # Distance ratio gradient (approximate)
    d_sq = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = G[i, i] + G[j, j] - 2 * G[i, j]
            d_sq[i, j] = val
            d_sq[j, i] = val
    
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            if d_sq[i, j] > 1e-12:
                off_diag.append((i, j, d_sq[i, j]))
    
    if not off_diag:
        return (psd_grad + rank_grad).flatten()
    
    min_d_sq = min(v for _, _, v in off_diag)
    max_d_sq = max(v for _, _, v in off_diag)
    
    if min_d_sq < 1e-12:
        return (psd_grad + rank_grad).flatten()
    
    ratio = max_d_sq / min_d_sq
    
    # Gradient of ratio = (max/min) w.r.t. G
    # d(ratio)/dG_ij = (1/min) * d(max)/dG_ij - (max/min^2) * d(min)/dG_ij
    # For max: d(max)/dG_ij = -2 if (i,j) is the max pair, else 0
    # For min: d(min)/dG_ij = -2 if (i,j) is the min pair, else 0
    
    ratio_grad = np.zeros((n, n))
    
    # Find max and min pairs
    max_pair = max(off_diag, key=lambda x: x[2])
    min_pair = min(off_diag, key=lambda x: x[2])
    
    i_max, j_max, _ = max_pair
    i_min, j_min, _ = min_pair
    
    # d(max_d_sq)/dG_ij = -2 for (i_max, j_max)
    ratio_grad[i_max, j_max] += (1.0 / min_d_sq) * (-2.0)
    ratio_grad[j_max, i_max] += (1.0 / min_d_sq) * (-2.0)
    
    # d(min_d_sq)/dG_ij = -2 for (i_min, j_min)
    ratio_grad[i_min, j_min] += (-max_d_sq / (min_d_sq ** 2)) * (-2.0)
    ratio_grad[j_min, i_min] += (-max_d_sq / (min_d_sq ** 2)) * (-2.0)
    
    # Symmetrize
    ratio_grad = (ratio_grad + ratio_grad.T) / 2.0
    
    total_grad = psd_grad + rank_grad + ratio_grad
    return total_grad.flatten()

def construct():
    n = N
    d = D
    start_time = time.time()
    
    # Initial Gram matrix: identity-like (points on unit sphere)
    np.random.seed(42)
    
    # Start with random points on sphere, compute Gram
    pts_init = np.random.randn(n, d)
    pts_init /= np.linalg.norm(pts_init, axis=1, keepdims=True)
    G_init = pts_init @ pts_init.T
    G_init = (G_init + G_init.T) / 2.0
    
    x0 = G_init.flatten()
    
    # Penalty parameters
    lambda_psd = 10.0
    lambda_rank = 1.0
    
    # Optimize using L-BFGS-B with gradient
    best_val = 1e10
    best_x = x0.copy()
    
    for trial in range(3):
        # Vary penalties
        lp = lambda_psd * (0.5 ** trial)
        lr = lambda_rank * (0.5 ** trial)
        
        try:
            res = minimize(
                _gram_objective, x0,
                jac=_gram_grad,
                args=(n, d, lp, lr),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception as e:
            pass
        
        if time.time() - start_time > 90:
            break
    
    # Extract points from best Gram matrix
    G_best = best_x.reshape(n, n)
    G_best = (G_best + G_best.T) / 2.0
    
    points = _gram_to_points(G_best, d)
    
    # Verify score
    score = _compute_score(points)
    
    if score > 1e8:
        # Fallback to simple random points
        pts_fallback = np.random.randn(n, d)
        return pts_fallback
    
    return points

# EVOLVE-BLOCK-END
