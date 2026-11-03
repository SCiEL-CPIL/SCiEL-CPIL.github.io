"""Improved packing: local optimization with soft-min objective and constraint handling."""
import numpy as np
import time
import sys

N = 50

def _softmin(x, k=1000.0):
    """Approximate min via log-sum-exp trick."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.inf
    m = np.min(x)
    return m - np.log(np.sum(np.exp(k * (x - m)))) / k

def _compute_score(centers):
    """Compute the packing radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if not (np.all(centers >= -0.5) and np.all(centers <= 0.5)):
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls
    wall_dists = np.minimum(0.5 - centers, centers + 0.5)
    min_wall = np.min(wall_dists)
    r = min(min_pair_dist / 2.0, min_wall)
    return float(r)

def _objective(centers, soft_k=50.0):
    """
    Objective to minimize: -min(pairwise_dist/2, wall_dist) approximated smoothly.
    We want to maximize r = min(min_pair/2, min_wall).
    So we minimize -r. Use softmin for smoothness.
    """
    centers = np.asarray(centers, dtype=np.float64)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    # Use squared distances for smoothness, then take sqrt at the end
    # min_pair_dist = sqrt(min_pair_dist_sq)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq)
    
    wall_dists = np.minimum(0.5 - centers, centers + 0.5)
    min_wall = np.min(wall_dists)
    
    # Soft min of (min_pair/2, min_wall)
    vals = np.array([min_pair / 2.0, min_wall])
    r_soft = _softmin(vals, k=soft_k)
    return -r_soft

def _grad_objective(centers, soft_k=50.0):
    """
    Gradient of the objective (negative soft-min radius).
    This is approximate; we use finite differences as a fallback if needed,
    but let's try analytical-ish.
    Actually, let's just use finite differences for reliability given time constraints.
    """
    # Finite difference gradient
    grad = np.zeros_like(centers, dtype=np.float64)
    h = 1e-6
    f0 = _objective(centers, soft_k)
    for i in range(N):
        for j in range(3):
            orig = centers[i, j]
            centers[i, j] = orig + h
            f1 = _objective(centers, soft_k)
            centers[i, j] = orig - h
            f2 = _objective(centers, soft_k)
            centers[i, j] = orig
            grad[i, j] = (f1 - f2) / (2 * h)
    return grad

def _project_to_cube(centers, margin=0.0):
    """Project centers to stay within the cube with a small margin."""
    lo = -0.5 + margin
    hi = 0.5 - margin
    return np.clip(centers, lo, hi)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with a grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Local optimization using gradient descent with adaptive step
    lr = 0.001
    soft_k = 50.0
    
    # Try a few iterations of gradient descent
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        grad = _grad_objective(centers, soft_k)
        centers = centers - lr * grad
        centers = _project_to_cube(centers, margin=0.01)
        
        # Occasionally evaluate true score
        if it % 100 == 0:
            score = _compute_score(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
    
    # Final evaluation
    final_score = _compute_score(centers)
    if final_score > best_score:
        best_score = final_score
        best_centers = centers.copy()
    
    return best_centers

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
