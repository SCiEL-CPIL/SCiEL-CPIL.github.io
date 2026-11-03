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

def _softmin_and_grad(centers, beta):
    """
    Compute smooth softmin of all constraint distances and its gradient.
    The score is min(min_pair/2, min_wall).
    We use softmin to get a smooth surrogate.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    
    # Compute all pairwise distances / 2
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    pair_dists = d / 2.0  # shape (n, n)
    
    # Wall distances: distance from each center to the nearest face
    # For center c_i, wall distance = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
    abs_c = np.abs(centers)
    wall_dists = 0.5 - np.max(abs_c, axis=1)  # shape (n,)
    
    # Combine all distances into one vector for softmin
    iu = np.triu_indices(n, k=1)
    pair_flat = pair_dists[iu]  # shape (n*(n-1)/2,)
    all_dists = np.concatenate([pair_flat, wall_dists])  # shape (n*(n-1)/2 + n,)
    
    # Softmin: -1/beta * log(sum(exp(-beta * D)))
    neg_beta_d = -beta * all_dists
    neg_beta_d -= np.max(neg_beta_d)  # for numerical stability
    exp_neg = np.exp(neg_beta_d)
    sum_exp = np.sum(exp_neg)
    softmin_val = -np.log(sum_exp) / beta
    
    # Weights for the softmin
    weights = exp_neg / sum_exp
    
    # Gradient computation
    grad = np.zeros_like(centers)
    
    # Pairwise contribution
    # For pair (i,j), D_ij = |c_i - c_j| / 2
    # dD_ij/dc_i = (1/2) * (c_i - c_j) / |c_i - c_j|
    
    pair_weights = np.zeros((n, n))
    pair_weights[iu] = weights[:len(pair_flat)]
    pair_weights = pair_weights + pair_weights.T  # symmetric
    
    # Avoid division by zero
    d_safe = np.where(d < 1e-12, 1e-12, d)
    
    # diff[i,j] = (c_i - c_j) / |c_i - c_j|
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    diff = diff / d_safe[:, :, np.newaxis]
    
    # grad_pair[i] = (1/2) * sum_j pair_weights[i,j] * diff[i,j]
    grad_pair = 0.5 * np.sum(pair_weights[:, :, np.newaxis] * diff, axis=1)
    
    # Wall contribution
    # For wall distance w_i = 0.5 - max_k |c_ik|
    # dw_i/dc_ik = -sign(c_ik) if k is argmax, 0 otherwise
    
    wall_weights = weights[len(pair_flat):]  # shape (n,)
    
    grad_wall = np.zeros_like(centers)
    max_idx = np.argmax(abs_c, axis=1)  # shape (n,)
    
    for i in range(n):
        k = max_idx[i]
        if abs_c[i, k] > 1e-12:
            grad_wall[i, k] = -np.sign(centers[i, k]) * wall_weights[i]
    
    grad = grad_pair + grad_wall
    
    return softmin_val, grad


def _optimize_positions(centers, max_iters=3000, lr_init=0.003, beta_start=30.0, beta_end=150.0):
    """
    Optimize positions using gradient ascent on softmin surrogate.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    lr = lr_init
    start_time = time.time()
    deadline = start_time + 90  # Leave margin
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        beta = beta_start + (beta_end - beta_start) * (it / max_iters)
        val, grad = _softmin_and_grad(centers, beta)
        
        # Gradient ascent with clipping
        centers = centers + lr * grad
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Evaluate actual score periodically
        if it % 10 == 0:
            score = _score(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
        
        # Learning rate decay
        lr *= 0.997
    
    return best_centers, best_score


def _increase_radius(centers, max_iters=2000, lr_init=0.001):
    """
    After positions are optimized, try to increase the effective radius
    by adjusting positions to maximize the minimum constraint.
    This is a second phase that focuses on the tightest constraints.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    lr = lr_init
    beta = 100.0
    start_time = time.time()
    deadline = start_time + 60
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        val, grad = _softmin_and_grad(centers, beta)
        
        centers = centers + lr * grad
        centers = np.clip(centers, -0.4999, 0.4999)
        
        if it % 10 == 0:
            score = _score(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
        
        lr *= 0.995
        beta = min(200.0, beta * 1.02)
    
    return best_centers, best_score


def _bcc_init():
    """
    Generate a BCC-like initial configuration for 100 spheres.
    BCC has better packing density than simple cubic.
    """
    # Start with a simple grid and add BCC-like perturbations
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    
    # Add small random perturbations to break symmetry
    np.random.seed(SEED)
    perturbed = grid + np.random.randn(N, 3) * 0.01
    
    return perturbed


def _random_init():
    """Generate a random initial configuration."""
    np.random.seed(SEED + 1)
    centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
    return centers


def construct():
    """
    Construct the centers of 100 equal spheres in a unit cube.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    best_centers = None
    best_score = 0.0
    
    # Try multiple initializations
    initializations = [
        ("bcc", _bcc_init()),
        ("random", _random_init()),
    ]
    
    # Add perturbed grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    initializations.append(("grid", grid.copy()))
    
    for name, start in initializations:
        if time.time() > deadline - 30:
            break
        
        # Phase 1: Optimize positions
        centers, score = _optimize_positions(start, max_iters=2000, lr_init=0.003,
                                             beta_start=30.0, beta_end=120.0)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
        
        # Phase 2: Increase radius (fine-tune)
        if time.time() < deadline - 20:
            centers2, score2 = _increase_radius(centers, max_iters=1500, lr_init=0.001)
            
            if score2 > best_score:
                best_score = score2
                best_centers = centers2.copy()
    
    # Final polish: very fine optimization
    if best_centers is not None and time.time() < deadline - 5:
        centers = best_centers.copy()
        lr = 0.0003
        beta = 150.0
        for it in range(500):
            if time.time() > deadline - 2:
                break
            val, grad = _softmin_and_grad(centers, beta)
            centers = centers + lr * grad
            centers = np.clip(centers, -0.4999, 0.4999)
            if it % 20 == 0:
                score = _score(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
            lr *= 0.99
    
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        return grid
    
    return best_centers
# EVOLVE-BLOCK-END
