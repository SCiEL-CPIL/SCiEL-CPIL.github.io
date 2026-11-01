"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 115.0  # Safety margin before 120s limit

def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and container margin."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Container constraint: 1 - |c_i|
    dists_origin = np.linalg.norm(points, axis=1)
    container_margin = 1.0 - np.max(dists_origin)
    if container_margin < 0:
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    return min(min_pair_dist, container_margin)

def _soft_min_obj_and_grad(points, beta=100.0):
    """
    Smooth surrogate: maximize soft-min of (pairwise_dists/2, container_margins).
    We minimize the negative soft-min.
    soft_min(x) = -(1/beta) * log(sum(exp(-beta * x_i)))
    """
    # Container margins
    dists_origin = np.linalg.norm(points, axis=1)
    container_margins = 1.0 - dists_origin
    
    # Pairwise distances / 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_margins = dists / 2.0
    
    # Combine all margins into one vector
    all_margins = np.concatenate([container_margins, pair_margins.ravel()])
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * x)))
    # For numerical stability, shift by min
    min_val = np.min(all_margins)
    shifted = -beta * (all_margins - min_val)
    log_sum = min_val - (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    
    # Objective to minimize: -log_sum
    obj = -log_sum
    
    # Gradient with respect to points
    # d/dx_i [ -(1/beta) log(sum_j exp(-beta * x_j)) ] = (sum_j exp(-beta*x_j) * (-beta) * dx_j/dx_i) / (beta * sum_j exp(-beta*x_j))
    # = - (sum_j exp(-beta*x_j) * dx_j/dx_i) / sum_j exp(-beta*x_j)
    # Let w_j = exp(-beta * (x_j - min_val)) / sum_k exp(-beta * (x_k - min_val))
    # Then grad = -sum_j w_j * (dx_j/dx_i)
    
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(points)
    
    # Container margins: x_j = 1 - ||p_j||, dx_j/dp_i = -p_j/||p_j|| if i==j
    for i in range(N):
        r = np.linalg.norm(points[i])
        if r > 1e-12:
            grad[i] += -weights[i] * (-points[i] / r)
    
    # Pairwise margins: x_{ij} = ||p_i - p_j||/2
    # dx_{ij}/dp_i = (p_i - p_j)/||p_i - p_j||, dx_{ij}/dp_j = (p_j - p_i)/||p_i - p_j||
    # Note: each pair appears twice in the flattened array (i,j) and (j,i)
    # But we filled diagonal with inf, so we need to handle carefully.
    # Actually, we flattened the upper triangle? No, we flattened the whole matrix with inf diagonal.
    # Let's restructure: only use i<j pairs to avoid double counting.
    
    # Let me redo this more carefully.
    # Reconstruct: we have N container margins and N*(N-1) pair margins (both i,j and j,i)
    # This is inefficient. Let me use only i<j pairs.
    
    # Actually, let's just compute the gradient numerically or use a cleaner formulation.
    # For speed, let's use a simpler approach: compute gradient via finite differences for the polish,
    # but for gradient ascent, let's use a vectorized approach.
    
    # Let me restart with a cleaner implementation.
    # We'll use only i<j pairs.
    
    # Recompute
    container_margins = 1.0 - np.linalg.norm(points, axis=1)
    
    # Pairwise: only i<j
    n_pairs = 0
    pair_indices = []
    for i in range(N):
        for j in range(i+1, N):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    
    diffs = points[pair_indices[:, 0]] - points[pair_indices[:, 1]]
    dists = np.linalg.norm(diffs, axis=1)
    pair_margins = dists / 2.0
    
    all_margins = np.concatenate([container_margins, pair_margins])
    n_total = len(all_margins)
    
    min_val = np.min(all_margins)
    shifted = -beta * (all_margins - min_val)
    log_sum = min_val - (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    obj = -log_sum
    
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(points)
    
    # Container part
    w_container = weights[:N]
    for i in range(N):
        r = np.linalg.norm(points[i])
        if r > 1e-12:
            grad[i] += w_container[i] * (points[i] / r)
    
    # Pair part
    w_pairs = weights[N:]
    for k, (i, j) in enumerate(pair_indices):
        d = diffs[k]
        dist = dists[k]
        if dist > 1e-12:
            dir_ij = d / dist
            grad[i] += w_pairs[k] * dir_ij
            grad[j] += w_pairs[k] * (-dir_ij)
    
    return obj, grad

def _ring_init():
    """Ring-based initialization: 1 center + concentric rings."""
    points = np.zeros((N, 2))
    # Center
    points[0] = [0.0, 0.0]
    
    # Ring structure: 1, 6, 12, 18, 13 = 50
    ring_sizes = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    idx = 1
    for ring_size, radius in zip(ring_sizes[1:], ring_radii[1:]):
        angles = np.linspace(0, 2*np.pi, ring_size, endpoint=False)
        for a in angles:
            points[idx] = [radius * np.cos(a), radius * np.sin(a)]
            idx += 1
    
    return points

def _gradient_ascent(points, beta_start=50.0, beta_end=200.0, n_iter=2000, lr_start=0.01, lr_end=0.0001):
    """Gradient ascent on soft-min objective with annealed beta and learning rate."""
    for it in range(n_iter):
        t = it / max(n_iter - 1, 1)
        beta = beta_start + (beta_end - beta_start) * t
        lr = lr_start * (lr_end / lr_start) ** t
        
        obj, grad = _soft_min_obj_and_grad(points, beta)
        points = points + lr * grad
        
        # Project back: ensure points stay in unit disk
        dists = np.linalg.norm(points, axis=1)
        mask = dists > 0.999
        if np.any(mask):
            points[mask] *= 0.999 / dists[mask]
    
    return points

def _nelder_mead_polish(points, max_iter=5000):
    """Nelder-Mead polish on the exact (nonsmooth) objective."""
    def neg_score(x):
        pts = x.reshape(N, 2)
        return -_compute_score(pts)
    
    result = minimize(neg_score, points.ravel(), method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14})
    return result.x.reshape(N, 2)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with ring initialization
    points = _ring_init()
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Gradient ascent with soft-min
    points = _gradient_ascent(points, beta_start=50.0, beta_end=200.0, n_iter=1500, lr_start=0.005, lr_end=0.00005)
    score = _compute_score(points)
    if score > best_score:
        best_points = points.copy()
        best_score = score
    
    # Nelder-Mead polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        polished = _nelder_mead_polish(points, max_iter=3000)
        score = _compute_score(polished)
        if score > best_score:
            best_points = polished.copy()
            best_score = score
    
    # Additional polish if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        polished = _nelder_mead_polish(best_points, max_iter=2000)
        score = _compute_score(polished)
        if score > best_score:
            best_points = polished.copy()
            best_score = score
    
    return best_points
# EVOLVE-BLOCK-END
