"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100
SEED = 42

# EVOLVE-BLOCK-START
def _score(centers):
    """Compute the score: largest common radius."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    # Check centers are inside the cube
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # Distance to walls
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dist)
    r = min(min_pair / 2.0, min_wall)
    return r

def _energy_and_grad(centers, r):
    """
    Compute a smooth surrogate energy that penalizes violations of:
      |c_i - c_j| >= 2r  for all i < j
      0.5 - |c_ik| >= r  for all i, k
    We use a soft-min approach: the objective is to maximize r,
    which is equivalent to minimizing a penalty that grows when constraints are violated.
    
    For the optimization, we use a potential-based approach:
    E = sum_{i<j} max(0, 2r - |c_i - c_j|)^2 + sum_{i,k} max(0, r - (0.5 - |c_ik|))^2
    This is non-smooth, so we use a smooth approximation.
    
    Actually, let's use a different strategy: we'll directly optimize for maximizing r
    using a penalty method. The score is min(min_pair/2, min_wall).
    
    We'll use a smooth surrogate: for each pair, define
    f_ij = 2r - |c_i - c_j|
    and for walls:
    g_ik = r - (0.5 - |c_ik|)
    
    The constraint is f_ij <= 0 and g_ik <= 0.
    
    We use a smooth max: softmax(x) = log(sum(exp(x_i))) as a smooth upper bound on max(x_i).
    But we need the minimum of the distances, which is a min, not max.
    
    Let's think differently. We want to maximize r such that all constraints hold.
    This is equivalent to: for a given configuration, r = min(min_pair/2, min_wall).
    
    For optimization, we can use the following smooth surrogate for the score:
    score_smooth = -logsumexp(-2 * distances_to_boundary_and_pairs) / 2
    where distances_to_boundary_and_pairs includes all pairwise distances (divided by 2)
    and all wall distances.
    
    Actually, let's use a simpler approach: we'll use gradient ascent on r with
    a penalty for constraint violations, or we'll use a direct optimization approach.
    
    Let me use a well-known approach for sphere packing: 
    - Start with a random or grid configuration
    - Use gradient-based optimization to maximize the minimum distance
    - The gradient of the minimum distance is tricky, so we use a smooth approximation
    
    I'll use the following: define D as the matrix of all relevant distances
    (pairwise/2 and wall distances). The score is min(D).
    
    A smooth approximation to min(D) is:
    softmin(D, beta) = -1/beta * log(sum(exp(-beta * D)))
    
    As beta -> infinity, this approaches min(D).
    
    The gradient of softmin with respect to center c_i is:
    d(softmin)/dc_i = -1/beta * sum_j exp(-beta * D_j) * dD_j/dc_i / sum_j exp(-beta * D_j)
    
    Let's implement this.
    """
    beta = 50.0  # Temperature for softmin
    
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    
    # Compute all pairwise distances / 2
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    pair_dists = d / 2.0  # shape (n, n)
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)  # shape (n,)
    
    # Combine all distances into one vector
    # We'll work with the flattened pairwise (upper triangle) and wall distances
    iu = np.triu_indices(n, k=1)
    pair_flat = pair_dists[iu]  # shape (n*(n-1)/2,)
    all_dists = np.concatenate([pair_flat, wall_dists])  # shape (n*(n-1)/2 + n,)
    
    # Softmin
    neg_beta_d = -beta * all_dists
    # For numerical stability
    neg_beta_d -= np.max(neg_beta_d)
    exp_neg = np.exp(neg_beta_d)
    log_sum = np.log(np.sum(exp_neg))
    softmin_val = -log_sum / beta
    
    # Weights for the softmin
    weights = exp_neg / np.sum(exp_neg)
    
    # Gradient with respect to centers
    # d(softmin)/dc_i = -1/beta * sum_j w_j * dD_j/dc_i
    # But note: softmin = -1/beta * log(sum(exp(-beta*D)))
    # d(softmin)/dc_i = -1/beta * (1/sum(exp(-beta*D))) * sum_j exp(-beta*D_j) * (-beta) * dD_j/dc_i
    #                  = sum_j w_j * dD_j/dc_i
    
    grad = np.zeros_like(centers)
    
    # Pairwise contribution
    # For pair (i, j), D_ij = |c_i - c_j| / 2
    # dD_ij/dc_i = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (2 * D_ij * 2) ... wait
    # |c_i - c_j| = 2 * D_ij, so dD_ij/dc_i = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (4 * D_ij)
    # Actually: D_ij = |c_i - c_j| / 2
    # dD_ij/dc_i = (1/2) * (c_i - c_j) / |c_i - c_j|
    
    # We need to accumulate: grad[i] += sum_j w_ij * dD_ij/dc_i
    # where w_ij is the weight for the pair (i,j) in the softmin
    
    # The weights are stored in pair_flat, indexed by the upper triangle
    # Let's create a weight matrix for pairs
    pair_weights = np.zeros((n, n))
    pair_weights[iu] = weights[:len(pair_flat)]
    pair_weights = pair_weights + pair_weights.T  # symmetric
    
    # For each pair (i,j) with i != j:
    # dD_ij/dc_i = (1/2) * (c_i - c_j) / |c_i - c_j|
    # dD_ij/dc_j = (1/2) * (c_j - c_i) / |c_i - c_j| = -dD_ij/dc_i
    
    # We can compute this as:
    # grad_pair[i] = sum_j pair_weights[i,j] * (1/2) * (c_i - c_j) / |c_i - c_j|
    
    # To avoid division by zero, add a small epsilon
    d_safe = np.where(d < 1e-12, 1e-12, d)
    
    # For each i, grad_pair[i] = (1/2) * sum_j pair_weights[i,j] * (c_i - c_j) / d_safe[i,j]
    # This can be computed efficiently
    
    # Create matrix of (c_i - c_j) / d_safe[i,j]
    # diff[i,j] = c_i - c_j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, n, 3)
    diff = diff / d_safe[:, :, np.newaxis]  # normalize
    
    # grad_pair[i] = (1/2) * sum_j pair_weights[i,j] * diff[i,j]
    # But pair_weights[i,i] = 0, so it's fine
    grad_pair = 0.5 * np.sum(pair_weights[:, :, np.newaxis] * diff, axis=1)
    
    # Wall contribution
    # For wall distance g_ik = 0.5 - |c_ik|, where |c_ik| is the max component
    # Actually, g_i = 0.5 - max_k |c_ik|
    # dg_i/dc_ik = -sign(c_ik) if k is the argmax, 0 otherwise
    
    wall_weights = weights[len(pair_flat):]  # shape (n,)
    
    # For each i, find the component with max |c_ik|
    abs_c = np.abs(centers)
    max_idx = np.argmax(abs_c, axis=1)  # shape (n,)
    
    grad_wall = np.zeros_like(centers)
    for i in range(n):
        k = max_idx[i]
        # dg_i/dc_ik = -sign(c_ik)
        grad_wall[i, k] = -np.sign(centers[i, k]) * wall_weights[i]
    
    grad = grad_pair + grad_wall
    
    # The gradient of the softmin is what we want to ascend
    # (since we want to maximize the softmin, which approximates the score)
    
    return softmin_val, grad


def _optimize_from_start(start_centers, max_iters=5000, lr_init=0.001, beta_schedule=True):
    """
    Optimize from a starting configuration using gradient ascent on the softmin surrogate.
    """
    centers = start_centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    lr = lr_init
    beta = 50.0
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin for 120s limit
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        val, grad = _energy_and_grad(centers, beta)
        
        # Gradient ascent
        centers = centers + lr * grad
        
        # Project back into the cube (soft constraint)
        # We'll clip to be inside the cube with a small margin
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Evaluate actual score periodically
        if it % 10 == 0:
            score = _score(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
        
        # Learning rate decay
        lr *= 0.995
        
        # Increase beta to make softmin sharper
        if beta_schedule:
            beta = min(200.0, beta * 1.01)
    
    return best_centers, best_score


def construct():
    """
    Construct the centers of 100 equal spheres in a unit cube.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    np.random.seed(SEED)
    
    best_centers = None
    best_score = 0.0
    
    # Start with a simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    
    # Try several random perturbations of the grid
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            start = grid.copy()
        else:
            # Add random perturbation
            start = grid + np.random.randn(N, 3) * 0.02
        
        centers, score = _optimize_from_start(start, max_iters=3000, lr_init=0.002)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
        
        if time.time() > deadline:
            break
    
    # Final polish: try a few more optimization steps with smaller learning rate
    if best_centers is not None and time.time() < deadline - 5:
        # Do a final optimization pass
        centers = best_centers.copy()
        lr = 0.0005
        beta = 100.0
        for it in range(1000):
            if time.time() > deadline - 2:
                break
            val, grad = _energy_and_grad(centers, beta)
            centers = centers + lr * grad
            centers = np.clip(centers, -0.4999, 0.4999)
            if it % 50 == 0:
                score = _score(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
            lr *= 0.99
    
    if best_centers is None:
        # Fallback to grid
        return grid
    
    return best_centers
# EVOLVE-BLOCK-END
